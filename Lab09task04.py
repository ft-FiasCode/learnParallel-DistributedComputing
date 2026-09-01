import pyopencl as cl
import numpy as np

# Input arrays
A = np.array([2, 4, 6, 8], dtype=np.float32)
B = np.array([1, 3, 5, 7], dtype=np.float32)

# Context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Buffers
A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
Temp_buf = cl.Buffer(ctx, mf.WRITE_ONLY, A.nbytes)

# Kernel
kernel_code = """
__kernel void vec_mul_temp(__global const float *A,
                           __global const float *B,
                           __global float *Temp) {
    int i = get_global_id(0);
    Temp[i] = A[i] * B[i];
}
"""
prg = cl.Program(ctx, kernel_code).build()

# Run kernel
prg.vec_mul_temp(queue, A.shape, None, A_buf, B_buf, Temp_buf)

# Copy result back
Temp = np.empty_like(A)
cl.enqueue_copy(queue, Temp, Temp_buf)

print("Temporary array:", Temp)
