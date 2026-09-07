import pyopencl as cl
import numpy as np
import time

N = 4
M = 4

A = np.random.randint(1, 10, size=(N, M)).astype(np.int32)
B = np.random.randint(1, 10, size=(N, M)).astype(np.int32)
C = np.empty_like(A)

kernel_code = """
__kernel void matrix_sub(__global const int* A,
                         __global const int* B,
                         __global int* C,
                         const int rows,
                         const int cols) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] - B[idx];
    }
}
"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)

mf = cl.mem_flags
buf_A = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
buf_B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
buf_C = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)

program = cl.Program(context, kernel_code).build()

start_time = time.time()
program.matrix_sub(queue, (N, M), None, buf_A, buf_B, buf_C, np.int32(N), np.int32(M))
cl.enqueue_copy(queue, C, buf_C).wait()
end_time = time.time()

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C (A - B):\n", C)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
