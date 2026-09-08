import pyopencl as cl
import numpy as np
import time

# Example: A is 4x2, B is 2x3
A = np.random.randint(1, 10, size=(4, 2)).astype(np.int32)
B = np.random.randint(1, 10, size=(2, 3)).astype(np.int32)
C = np.empty((A.shape[0], B.shape[1]), dtype=np.int32)

# Kernel for non-square matrix multiplication
kernel_code = """
__kernel void matmul(__global const int* A,
                     __global const int* B,
                     __global int* C,
                     const int rowsA,
                     const int colsA,
                     const int colsB) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}
"""

# OpenCL setup
context = cl.create_some_context()
queue = cl.CommandQueue(context)

mf = cl.mem_flags
buf_A = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
buf_B = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
buf_C = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)

program = cl.Program(context, kernel_code).build()

# Execute kernel
start_time = time.time()
program.matmul(queue, C.shape, None,
               buf_A, buf_B, buf_C,
               np.int32(A.shape[0]), np.int32(A.shape[1]), np.int32(B.shape[1]))
cl.enqueue_copy(queue, C, buf_C).wait()
end_time = time.time()

# Print results
print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C (A x B):\n", C)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
