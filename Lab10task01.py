import pyopencl as cl
import numpy as np
import time


input_str = "This Is PDC Lab 10"
data = np.frombuffer(input_str.encode('ascii'), dtype=np.uint8).copy()


count = np.zeros(1, dtype=np.int32)


kernel_code = """
__kernel void count_uppercase(__global const char* str,
                              __global int* count,
                              const int length) {
    int gid = get_global_id(0);
    if (gid < length) {
        char c = str[gid];
        if (c >= 'A' && c <= 'Z') {
            atomic_inc(count);
        }
    }
}
"""


context = cl.create_some_context()
queue = cl.CommandQueue(context)


mf = cl.mem_flags
str_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
count_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=count)


program = cl.Program(context, kernel_code).build()


start_time = time.time()


program.count_uppercase(queue, (data.shape[0],), None,
                        str_buf, count_buf, np.int32(data.shape[0]))


cl.enqueue_copy(queue, count, count_buf).wait()

end_time = time.time()


print("Input String:", input_str)
print("Number of uppercase letters:", count[0])
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
