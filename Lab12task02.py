import pyopencl as cl
import numpy as np

#  USER INPUT 
user_input = input("Enter integers separated by space: ")
arr = np.array(list(map(int, user_input.split())), dtype=np.int32)

n = len(arr)

#  OPENCL SETUP 
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OPENCL KERNELS 
kernel_code = """
__kernel void extract_bit(__global int* arr, __global int* bits, int bit)
{
    int i = get_global_id(0);
    bits[i] = (arr[i] >> bit) & 1;
}

__kernel void prefix_sum(__global int* bits, __global int* scan, int n)
{
    int i = get_global_id(0);

    int sum = 0;
    for(int j = 0; j <= i; j++)
        sum += bits[j];

    scan[i] = sum;
}

__kernel void reorder(__global int* input,
                      __global int* output,
                      __global int* bits,
                      __global int* scan,
                      int n)
{
    int i = get_global_id(0);

    int zero_count = n - scan[n-1];
    int pos;

    if(bits[i] == 0)
        pos = i - scan[i];
    else
        pos = zero_count + scan[i] - 1;

    output[pos] = input[i];
}
"""

program = cl.Program(context, kernel_code).build()

mf = cl.mem_flags

# Buffers
buf_input = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=arr)
buf_output = cl.Buffer(context, mf.READ_WRITE, arr.nbytes)
buf_bits = cl.Buffer(context, mf.READ_WRITE, arr.nbytes)
buf_scan = cl.Buffer(context, mf.READ_WRITE, arr.nbytes)

# RADIX SORT 
print("\nInitial Array:", arr)

for bit in range(32):  # 32-bit integers

    # Step 1: Extract bit
    program.extract_bit(queue, (n,), None, buf_input, buf_bits, np.int32(bit))

    # Step 2: Prefix sum
    program.prefix_sum(queue, (n,), None, buf_bits, buf_scan, np.int32(n))

    # Step 3: Reorder
    program.reorder(queue, (n,), None,
                    buf_input, buf_output, buf_bits, buf_scan, np.int32(n))

    queue.finish()

    # Copy back to host for visualization
    cl.enqueue_copy(queue, arr, buf_output)

    print(f"\nAfter processing bit {bit}:")
    print(arr)

    # Swap buffers for next iteration
    buf_input, buf_output = buf_output, buf_input

#  FINAL OUTPUT 
print("\nFinal Sorted Array:")
print(arr)