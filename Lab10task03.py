import pyopencl as cl
import numpy as np
import time


input_str = "racecar"
data = np.frombuffer(input_str.encode('ascii'), dtype=np.uint8).copy()


result = np.ones(1, dtype=np.int32)


kernel_code = """
__kernel void check_palindrome(__global const char* str,
                               __global int* result,
                               const int length) {
    int gid = get_global_id(0);
    if (gid < length / 2) {
        if (str[gid] != str[length - gid - 1]) {
            result[0] = 0; // mark as not palindrome
        }
    }
}
"""


context = cl.create_some_context()
queue = cl.CommandQueue(context)


mf = cl.mem_flags
str_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
result_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=result)


program = cl.Program(context, kernel_code).build()


start_time = time.time()


program.check_palindrome(queue, (data.shape[0] // 2,), None,
                         str_buf, result_buf, np.int32(data.shape[0]))


cl.enqueue_copy(queue, result, result_buf).wait()

end_time = time.time()


print("Input String:", input_str)
print("Result:", "Palindrome" if result[0] == 1 else "Not Palindrome")
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
