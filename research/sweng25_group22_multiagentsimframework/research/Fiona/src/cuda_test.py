from numba import cuda
import numpy as np
import os

print("Starting Program. PID=" + str(os.getpid()));
f = open("../data/PID.txt", "w");
f.write("PID=" + str(os.getpid()));
f.close();

# Define CUDA kernel
@cuda.jit
def add_arrays(x, y, out):
    idx = cuda.grid(1)  # Get thread index
    if idx < x.size:
        out[idx] = x[idx] + y[idx]

# Initialize data
n = 100000
x = np.arange(n, dtype=np.float32)
y = np.arange(n, dtype=np.float32)
out = np.zeros(n, dtype=np.float32)

# Allocate memory on the GPU
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.to_device(out)

# Define grid and block dimensions
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

# Launch CUDA kernel
add_arrays[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

# Copy result back to CPU
out = d_out.copy_to_host()

print(out[:100])  # Print first 10 elements
