# CS79995 Assignment 1 — Matrix Multiplication in CUDA

**Author:** Waylon Luo
**Course:** CS 79995 GPU Parallel Processing
**Instructor:** Qiang Guan

This program multiplies two square integer matrices on the GPU with CUDA,
then checks the result against a single-threaded CPU multiplication and
reports how long each one took.

## Files

| File | Description |
|------|-------------|
| [matrixMultiply.cu](matrixMultiply.cu) | CUDA source: the GPU kernel, the CPU check, and the timing code |
| [Report.txt](Report.txt) | Assignment report with the timing results |

## How it works

1. The program fills two `SIZE × SIZE` matrices with random 0s and 1s.
2. It copies both matrices to the GPU and launches the `MatrixMul` kernel.
   Each thread computes one element of the result as the dot product of a
   row and a column.
3. It copies the result back and runs `verify_result`, which repeats the
   multiplication on the CPU and `assert`s that every element matches.
4. Both steps are timed with CUDA events.

## Configuration

Set the two macros at the top of `matrixMultiply.cu`:

```c
#define SIZE 64     // matrices are SIZE x SIZE
#define THREADS 32  // each block is THREADS x THREADS threads
```

- The grid has `SIZE / THREADS` blocks in each direction, so `SIZE` must be
  divisible by `THREADS`. For example, with `SIZE 32` you can use `THREADS`
  4, 8, 16, or 32.
- A CUDA block can hold at most 1024 threads, so `THREADS` can be at most 32
  (32 × 32 = 1024).

## Build and run

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

## Results

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
