# CS79995-Assignment1
### Name - Waylon Luo
#### Instructor:  Qiang Guan 
#### CS 79995 GPU Parallel Process 
#### Assignment 1 - Matrix Multiplication 

Program File - matrixMultiply.cu

By assigning the SIZE and THREADS at the beginning of the program, 
one can adjust the size of the matrices. The number of SIZE/THREADS
should be an integer. For example, if the size of a matrix is 32 then, 
the number of threads should be 4, 8, or 16.. The comparison between 
GPU and CPU computing times are as followed. 

|Matrix Size    | THREADS | GPU time (ms) |	CPU time (ms) |
|---------------|---------|---------------|-------------------|
|	16	| 16	  |	0.048288  |	 0.035648     |
|	32	| 32	  |	0.039136  |	 0.225824     |
|	64	| 32	  |	0.04752	  |	 1.683584     |
|	128	| 32	  |	0.063936  |	 12.701888    |
|	256	| 32	  |	0.088352  |	 100.200546   |
