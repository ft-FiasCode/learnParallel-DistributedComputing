import pyopencl as cl
import numpy as np

# Example input vectors
A = np.array([5, 10, 15], dtype=np.float32)
B = np.array([2, 4, 6], dtype=np.float32)

# Handle size mismatch (truncate to min length)
min_len = min(len(A), len(B))
A = A[:min_len]
B = B[:min_len]

# Create output array
C = np.empty_like(A)

# OpenCL kernel for subtraction
kernel_code = """
__kernel void vector_sub(__global const float *A,
                         __global const float *B,
                         __global float *C) {
    int gid = get_global_id(0);
    C[gid] = A[gid] - B[gid];
}
"""

# Setup OpenCL
platform = cl.get_platforms()[0]          # First platform
device = platform.get_devices()[0]        # First device
context = cl.Context([device])
queue = cl.CommandQueue(context)

# Create buffers
mf = cl.mem_flags
buf_A = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
buf_B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
buf_C = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)

# Build and run kernel
program = cl.Program(context, kernel_code).build()
program.vector_sub(queue, A.shape, None, buf_A, buf_B, buf_C)

# Copy result back
cl.enqueue_copy(queue, C, buf_C)

# Display results
print("Vector A:", A)
print("Vector B:", B)
print("Result (A - B):", C)
