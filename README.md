# CS79995 Assignment 1 — Matrix Multiplication in CUDA

**Author:** Waylon Luo
**Course:** CS 79995 GPU Parallel Processing
**Instructor:** Qiang Guan

This project multiplies two square integer matrices on the GPU with CUDA,
checks the result against a single-threaded CPU multiplication, and reports
how long each one took.

There are two versions of the program:

- `matrixMultiply.cu` — the original, naive kernel (one global-memory load
  per operand per multiply-add).
- `matrixMultiplyOptimized.cu` — a shared-memory tiled kernel, plus the
  original kernel as a baseline so both can be timed in a single run.

## Files

| File | Description |
|------|-------------|
| [matrixMultiply.cu](matrixMultiply.cu) | Original CUDA source: naive GPU kernel, CPU check, and timing code |
| [matrixMultiplyOptimized.cu](matrixMultiplyOptimized.cu) | Optimized CUDA source: tiled kernel, naive baseline, CPU check, timing, error checking |
| [Report.txt](Report.txt) | Assignment report with the timing results for the original version |

## How it works (both versions)

1. The program fills two `SIZE × SIZE` matrices with random 0s and 1s.
2. It copies both matrices to the GPU and launches a matrix-multiply kernel.
   Each thread computes one element of the result as the dot product of a
   row of A and a column of B.
3. It copies the result back and repeats the multiplication on the CPU to
   verify that every element matches.
4. GPU work is timed with CUDA events.

## Original version: `matrixMultiply.cu`

### Configuration

Set the two macros at the top of the file:

```c
#define SIZE 64     // matrices are SIZE x SIZE
#define THREADS 32  // each block is THREADS x THREADS threads
```

- The grid has `SIZE / THREADS` blocks in each direction, so `SIZE` must be
  divisible by `THREADS`. For example, with `SIZE 32` you can use `THREADS`
  4, 8, 16, or 32.
- A CUDA block can hold at most 1024 threads, so `THREADS` can be at most 32
  (32 × 32 = 1024).

### Build and run

You need an NVIDIA GPU and the CUDA Toolkit.

```bash
nvcc -o matrixMultiply matrixMultiply.cu
./matrixMultiply
```

Sample output:

```
Matrix Size: 64 x 64
No. of blocks: 2 x 2
No. of therads: 32 x 32
Compute time on GPU:  0.047520 ms
Compute time on CPU:  1.683584 ms
```

### Results

| Matrix size | THREADS | GPU time (ms) | CPU time (ms) | CPU / GPU |
|------------:|--------:|--------------:|--------------:|----------:|
| 16  | 16 | 0.048288 | 0.035648   | 0.7×   |
| 32  | 32 | 0.039136 | 0.225824   | 5.8×   |
| 64  | 32 | 0.047520 | 1.683584   | 35×    |
| 128 | 32 | 0.063936 | 12.701888  | 199×   |
| 256 | 32 | 0.088352 | 100.200546 | 1134×  |

For a 16 × 16 matrix the CPU is faster, because launching a kernel costs
more than the arithmetic saves. As the matrix grows, CPU time rises roughly
8× each time the size doubles (O(N³)), while GPU time barely changes.

Notes on the timings:

- GPU time covers only the kernel. It does not include copying data between
  the host and the device.
- CPU time includes the `assert` on every element, so it is slightly higher
  than a plain multiplication would take.

## Optimized version: `matrixMultiplyOptimized.cu`

### Configuration

```c
#define SIZE 64   // matrices are SIZE x SIZE (any positive value)
#define TILE 32   // block is TILE x TILE threads; shared-memory tile is TILE x TILE
#define REPS 10   // timed launches per kernel; the average is reported
```

All three can also be overridden on the command line, so you do not have to
edit the file to sweep sizes:

```bash
nvcc -O3 -DSIZE=1024 -o matrixMultiplyOptimized matrixMultiplyOptimized.cu
```

Unlike the original, `SIZE` does **not** need to be a multiple of `TILE`.
The kernel pads out-of-range tile elements with zero and skips out-of-range
writes.

### Build and run

```bash
nvcc -O3 -o matrixMultiplyOptimized matrixMultiplyOptimized.cu
./matrixMultiplyOptimized
```

The program prints the average kernel time for the naive baseline and for
the tiled kernel, the CPU time, the effective GFLOP/s of each kernel, and
whether each GPU result matched the CPU. Example of the output format:

```
Matrix Size: 1024 x 1024
No. of blocks:  32 x 32
No. of threads: 32 x 32
Timed launches per kernel: 10 (plus 1 warm-up)

Compute time on GPU (naive):   x.xxxxxx ms  (  xxx.xx GFLOP/s)  OK
Compute time on GPU (tiled):   x.xxxxxx ms  (  xxx.xx GFLOP/s)  OK
Compute time on CPU:           x.xxxxxx ms

Speedup tiled vs naive: x.xxx
Speedup tiled vs CPU:   x.xxx
```

The optimized file has not yet been run on a GPU (it was written on a machine
without the CUDA Toolkit), so no measured numbers are recorded here. Fill in
the table below after running it:

| Matrix size | Naive GPU (ms) | Tiled GPU (ms) | CPU (ms) | Tiled / Naive speedup |
|------------:|---------------:|---------------:|---------:|----------------------:|
| 64   | | | | |
| 256  | | | | |
| 1024 | | | | |
| 2048 | | | | |

Expect little difference at 64 × 64, where launch overhead dominates and the
whole problem fits in L2 cache. The gap should open up from about 512 × 512
onward, where the naive kernel becomes bound by global-memory bandwidth.

## What changed and why

### 1. Shared-memory tiling (the main optimization)

**Original.** Each thread loops over `k` and reads `Md[row*W + k]` and
`Nd[k*W + col]` directly from global memory on every iteration. For an N × N
problem each thread issues 2N global loads, and the 32 × 32 threads in a
block collectively load each row of A and each column of B 32 times over.
The only reuse comes from the hardware caches, which the kernel does not
control.

**Optimized.** The block walks along the k dimension in steps of `TILE`. At
each step the 1024 threads cooperatively load one 32 × 32 tile of A and one
of B into `__shared__` arrays (one element per thread), synchronize, and
then each thread performs 32 multiply-adds reading only from shared memory.
Every element of A and B is now fetched from global memory once per block
instead of once per thread, cutting global traffic by a factor of `TILE`
(32×). Shared memory is on-chip and roughly 100× lower latency than DRAM, so
the inner loop is no longer memory-bound.

The loads are coalesced: consecutive `threadIdx.x` values read consecutive
addresses from both A (`A[row*N + t*TILE + tx]`) and B
(`B[(t*TILE + ty)*N + col]`), so each warp's load is a single contiguous
128-byte transaction. The shared-memory reads `As[ty][k]` (broadcast within
a warp) and `Bs[k][tx]` (consecutive banks) are bank-conflict free.

### 2. Accumulate in a register, write once

**Original.** The inner loop does `Pd[row*W + col] += ...`, which is a global
read-modify-write on every iteration: N loads and N stores of the output per
thread, on top of the operand loads.

**Optimized.** The running sum lives in a local `int sum` (a register) and is
stored to `C` exactly once after the loop. This removes 2N global memory
operations per thread.

### 3. Correctness fix: uninitialized output

**Original.** Because the kernel uses `+=`, it depends on `Pd` starting at
zero. But `cudaMalloc` does not zero memory, so the result is only correct
when the freshly allocated buffer happens to contain zeros. This usually
holds on a quiet GPU, which is why the `assert` passed, but it is undefined
behaviour in general.

**Optimized.** The tiled kernel assigns with `=` and never reads `C`. The
naive baseline is kept for comparison, and the host code `cudaMemset`s the
output to zero before every launch of it so its timing and result are valid.

### 4. Arbitrary matrix sizes

**Original.** `BLOCKS = N / THREADS` uses integer division, so if `SIZE` is
not a multiple of `THREADS` some rows and columns are never computed.

**Optimized.** The grid is `ceil(N / TILE)` blocks per dimension. Inside the
kernel, out-of-range tile elements are loaded as zero and out-of-range
results are not written, so any `SIZE` works.

### 5. Error checking

**Original.** Defines a `CUDA_CALL` macro but never uses it. If
`cudaMalloc`, `cudaMemcpy`, or the kernel launch fails, the program carries
on silently and reports a meaningless time.

**Optimized.** Every CUDA API call goes through `CUDA_CHECK`, which prints
the file, line, and `cudaGetErrorString` and exits. `cudaGetLastError` is
checked after each launch to catch invalid launch configurations.

### 6. More reliable timing

**Original.** Times a single, cold kernel launch. The first launch of a
kernel pays for module loading and context setup, which for small matrices
is larger than the kernel itself (that is why the 16 × 16 and 32 × 32 GPU
times in the table above are almost identical).

**Optimized.** Performs one untimed warm-up launch, then averages `REPS`
timed launches. It also reports GFLOP/s (2·N³ operations / time) so results
at different sizes can be compared directly, and prints the speedup of the
tiled kernel over both the naive kernel and the CPU.

### 7. Smaller host-side changes

- The CPU reference is timed with `std::chrono` instead of CUDA events, and
  verification is a plain comparison rather than an `assert`, so the program
  reports `OK`/`WRONG` and exits non-zero on a mismatch instead of aborting.
- Both GPU results (naive and tiled) are checked against the CPU result.
- `srand(0)` makes the random matrices reproducible between runs.
- `const __restrict__` on the kernel pointers tells the compiler the inputs
  and output do not alias, which lets it schedule loads more freely.

### Side-by-side summary

| Aspect | `matrixMultiply.cu` | `matrixMultiplyOptimized.cu` |
|--------|---------------------|------------------------------|
| Operand reads | Global memory, 2N per thread | Shared-memory tiles, 2N / TILE global loads per thread |
| Output accumulation | Global `+=` every iteration | Register, one store at the end |
| Output must be pre-zeroed | Yes (and it isn't) | No |
| `SIZE` not a multiple of block size | Silently wrong | Handled |
| CUDA error checking | None | Every call |
| Timing | One cold launch | Warm-up + average of `REPS` |
| Reported metrics | ms | ms, GFLOP/s, speedups |
| Verification | `assert` (aborts) | Compare, print `OK`/`WRONG`, non-zero exit |
