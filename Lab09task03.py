import pyopencl as cl
import numpy as np

# Input arrays
A = np.array([1, 2, 3, 4, 5], dtype=np.float32)
B = np.array([10, 20, 30, 40, 50], dtype=np.float32)

# Context, queue, and memory flags
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Buffers
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, A.nbytes)

# Kernel
kernel_code = """
__kernel void vec_mul(__global const float *A,
                      __global const float *B,
                      __global float *C) {
    int i = get_global_id(0);
    C[i] = A[i] * B[i];
}
"""
prg = cl.Program(ctx, kernel_code).build()

# Run kernel
prg.vec_mul(queue, A.shape, None, A_buf, B_buf, C_buf)

# Copy result back
C = np.empty_like(A)
cl.enqueue_copy(queue, C, C_buf)

print("Result:", C)
