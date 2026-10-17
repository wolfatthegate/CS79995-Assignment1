// matrixMultiplyOptimized.cu
//
// Optimized square matrix multiplication (C = A * B) in CUDA.
//
// This is a companion to matrixMultiply.cu. It keeps the same overall
// structure (random 0/1 integer matrices, GPU multiply, CPU check, timing)
// but replaces the kernel with a shared-memory tiled kernel and fixes a few
// correctness issues in the original. The original naive kernel is kept in
// this file as a baseline so both can be timed in a single run.
//
// Build:  nvcc -O3 -o matrixMultiplyOptimized matrixMultiplyOptimized.cu
// Run:    ./matrixMultiplyOptimized

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

using std::generate;
using std::vector;

// Matrices are SIZE x SIZE. Any positive value works; it no longer has to be
// a multiple of TILE because the kernels guard their bounds.
#ifndef SIZE
#define SIZE 64
#endif

// Each block is TILE x TILE threads and works on a TILE x TILE output tile.
// 32 is the largest value allowed (32 * 32 = 1024 threads per block) and it
// matches the warp width, so each row of a tile is loaded by one full warp.
#ifndef TILE
#define TILE 32
#endif

// Number of timed launches per kernel. The average is reported.
#ifndef REPS
#define REPS 10
#endif

// Abort with a message if a CUDA call fails. The original file defined a
// similar macro but never used it, so errors went unnoticed.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Baseline kernel (same algorithm as matrixMultiply.cu).
//
// Every thread computes one output element and reads its whole row of A and
// column of B straight from global memory. For an N x N problem that is
// 2 * N global loads per output element, and nothing is reused between
// threads except through the L1/L2 caches. It also accumulates into Pd with
// "+=", so Pd must be zeroed before the launch (the original did not do
// this; cudaMalloc does not clear memory).
// ---------------------------------------------------------------------------
__global__ void MatrixMulNaive(const int *Md, const int *Nd, int *Pd, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        for (int k = 0; k < Width; ++k) {
            Pd[row * Width + col] += Md[row * Width + k] * Nd[k * Width + col];
        }
    }
}

// ---------------------------------------------------------------------------
// Optimized kernel: shared-memory tiling.
//
// The block that owns output tile (by, bx) walks along the K dimension in
// steps of TILE. At each step the block cooperatively loads one TILE x TILE
// tile of A and one of B into shared memory (one element per thread), then
// every thread computes TILE multiply-adds from shared memory before the
// block moves on to the next tile.
//
// Each element of A and B is now loaded from global memory once per block
// instead of once per thread, so global traffic drops by a factor of TILE
// (32x). The running sum lives in a register and is written to C exactly
// once, instead of a global read-modify-write on every iteration.
//
// Bounds are guarded with zero padding, so N does not need to be a multiple
// of TILE. Memory access is coalesced: consecutive threadIdx.x values read
// consecutive addresses from both A and B.
// ---------------------------------------------------------------------------
__global__ void MatrixMulTiled(const int *__restrict__ A,
                               const int *__restrict__ B,
                               int *__restrict__ C, int N) {
    __shared__ int As[TILE][TILE];
    __shared__ int Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    int sum = 0;

    int numTiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + tx;   // column of A this thread loads
        int bRow = t * TILE + ty;   // row of B this thread loads

        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0;

        __syncthreads();   // tile fully loaded before anyone reads it

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // everyone done reading before the next load
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Launch `kernel` REPS times and return the average time per launch in ms.
// One untimed warm-up launch absorbs first-launch costs (module load,
// context setup) so the number reflects the kernel itself.
template <typename Kernel>
float timeKernel(Kernel kernel, dim3 grid, dim3 block, const int *dA,
                 const int *dB, int *dC, int N, size_t bytes, bool zeroOutput) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up
    if (zeroOutput) CUDA_CHECK(cudaMemset(dC, 0, bytes));
    kernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float total = 0.0f;
    for (int i = 0; i < REPS; ++i) {
        if (zeroOutput) CUDA_CHECK(cudaMemset(dC, 0, bytes));
        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / REPS;
}

// Single-threaded CPU reference. Returns elapsed ms.
double cpuMultiply(const vector<int> &a, const vector<int> &b, vector<int> &c, int N) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int tmp = 0;
            for (int k = 0; k < N; ++k) {
                tmp += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = tmp;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Compare two result matrices element by element.
bool matricesEqual(const vector<int> &x, const vector<int> &y, int N, const char *label) {
    for (int i = 0; i < N * N; ++i) {
        if (x[i] != y[i]) {
            fprintf(stderr, "MISMATCH (%s) at [%d][%d]: %d vs %d\n", label,
                    i / N, i % N, x[i], y[i]);
            return false;
        }
    }
    return true;
}

double gflops(int N, float ms) {
    return 2.0 * N * N * N / (ms * 1.0e6);
}

int main(void) {
    const int N = SIZE;
    const size_t bytes = (size_t)N * N * sizeof(int);

    printf("Matrix Size: %d x %d\n", N, N);

    vector<int> host_a(N * N), host_b(N * N);
    vector<int> host_naive(N * N), host_tiled(N * N), host_cpu(N * N);

    srand(0);
    generate(host_a.begin(), host_a.end(), []() { return rand() % 2; });
    generate(host_b.begin(), host_b.end(), []() { return rand() % 2; });

    int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_a, bytes));
    CUDA_CHECK(cudaMalloc(&dev_b, bytes));
    CUDA_CHECK(cudaMalloc(&dev_c, bytes));

    CUDA_CHECK(cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

    // Round up so every element is covered even when N % TILE != 0.
    int blocksPerDim = (N + TILE - 1) / TILE;
    dim3 block(TILE, TILE);
    dim3 grid(blocksPerDim, blocksPerDim);

    printf("No. of blocks:  %d x %d\n", blocksPerDim, blocksPerDim);
    printf("No. of threads: %d x %d\n", TILE, TILE);
    printf("Timed launches per kernel: %d (plus 1 warm-up)\n\n", REPS);

    // --- baseline (naive) kernel ---
    float naiveMs = timeKernel(MatrixMulNaive, grid, block, dev_a, dev_b, dev_c,
                               N, bytes, /*zeroOutput=*/true);
    CUDA_CHECK(cudaMemcpy(host_naive.data(), dev_c, bytes, cudaMemcpyDeviceToHost));

    // --- optimized (tiled) kernel ---
    float tiledMs = timeKernel(MatrixMulTiled, grid, block, dev_a, dev_b, dev_c,
                               N, bytes, /*zeroOutput=*/false);
    CUDA_CHECK(cudaMemcpy(host_tiled.data(), dev_c, bytes, cudaMemcpyDeviceToHost));

    // --- CPU reference ---
    double cpuMs = cpuMultiply(host_a, host_b, host_cpu, N);

    bool okNaive = matricesEqual(host_naive, host_cpu, N, "naive vs CPU");
    bool okTiled = matricesEqual(host_tiled, host_cpu, N, "tiled vs CPU");

    printf("Compute time on GPU (naive): %10.6f ms  (%8.2f GFLOP/s)  %s\n",
           naiveMs, gflops(N, naiveMs), okNaive ? "OK" : "WRONG");
    printf("Compute time on GPU (tiled): %10.6f ms  (%8.2f GFLOP/s)  %s\n",
           tiledMs, gflops(N, tiledMs), okTiled ? "OK" : "WRONG");
    printf("Compute time on CPU:         %10.6f ms\n", cpuMs);
    printf("\nSpeedup tiled vs naive: %.2fx\n", naiveMs / tiledMs);
    printf("Speedup tiled vs CPU:   %.2fx\n", cpuMs / tiledMs);

    CUDA_CHECK(cudaFree(dev_a));
    CUDA_CHECK(cudaFree(dev_b));
    CUDA_CHECK(cudaFree(dev_c));

    return (okNaive && okTiled) ? EXIT_SUCCESS : EXIT_FAILURE;
}
