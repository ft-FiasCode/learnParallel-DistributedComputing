import pyopencl as cl
import numpy as np
import time


input_str = "Lab 10 task 02 aeiou "
data = np.frombuffer(input_str.encode('ascii'), dtype=np.uint8).copy()


kernel_code = """
__kernel void replace_vowels(__global char* str, const int length) {
    int gid = get_global_id(0);
    if (gid < length) {
        char c = str[gid];
        if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||
            c=='A'||c=='E'||c=='I'||c=='O'||c=='U') {
            str[gid] = '*';
        }
    }
}
"""


context = cl.create_some_context()
queue = cl.CommandQueue(context)


mf = cl.mem_flags
str_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=data)


program = cl.Program(context, kernel_code).build()


start_time = time.time()


program.replace_vowels(queue, (data.shape[0],), None, str_buf, np.int32(data.shape[0]))


cl.enqueue_copy(queue, data, str_buf).wait()

end_time = time.time()


output_str = data.tobytes().decode('ascii')
print("Original String:", input_str)
print("Modified String:", output_str)
print("Execution Time: {:.6f} seconds".format(end_time - start_time))
