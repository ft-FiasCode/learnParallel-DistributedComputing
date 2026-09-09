import pyopencl as cl
import numpy as np
import time

A = np.random.randint(1, 10, size=(3, 4)).astype(np.int32)
T = np.empty((A.shape[1], A.shape[0]), dtype=np.int32)  # Transposed matrix

kernel_code = """
__kernel void transpose(__global const int* A,
                        __global int* T,
                        const int rows,
                        const int cols) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < rows && col < cols) {
        int idxA = row * cols + col;
        int idxT = col * rows + row;
        T[idxT] = A[idxA];
    }
}
"""

context = cl.create_some_context()
queue = cl.CommandQueue(context)

mf = cl.mem_flags
buf_A = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
buf_T = cl.Buffer(context, mf.WRITE_ONLY, T.nbytes)

program = cl.Program(context, kernel_code).build()

start_time = time.time()
program.transpose(queue, A.shape, None,
                  buf_A, buf_T,
                  np.int32(A.shape[0]), np.int32(A.shape[1]))
cl.enqueue_copy(queue, T, buf_T).wait()
end_time = time.time()

print("Original Matrix A:\n", A)
print("Transposed Matrix T:\n", T)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
