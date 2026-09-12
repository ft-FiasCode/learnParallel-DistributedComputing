import pyopencl as cl
import numpy as np
import time

# CPU Heap Sort 

def cpu_heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        cpu_heapify(arr, n, largest)

def cpu_heap_sort(arr):
    n = len(arr)
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        cpu_heapify(arr, n, i)
    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        cpu_heapify(arr, i, 0)
    return arr

# GPU Heapify Kernel (OpenCL)

kernel_code = """
__kernel void heapify(__global int* arr, int N) {
    int i = get_global_id(0);
    int left = 2*i + 1;
    int right = 2*i + 2;
    int largest = i;

    if (left < N && arr[left] > arr[largest])
        largest = left;
    if (right < N && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
    }
}
"""

# GPU Heap Sort 

def gpu_heap_sort(arr):
    N = len(arr)
    arr_np = np.array(arr, dtype=np.int32)

    # Setup OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Create buffer
    d_array = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr_np)

    # Build program
    program = cl.Program(context, kernel_code).build()
    kernel = program.heapify

    # Run heapify multiple times (simplified)
    for i in range(N // 2, -1, -1):
        kernel(queue, (N,), None, d_array, np.int32(N))

    # Copy back
    result = np.empty_like(arr_np)
    cl.enqueue_copy(queue, result, d_array).wait()

    return result


if __name__ == "__main__":
    # Generate random array
    N = 20
    arr = np.random.randint(0, 100, size=N).tolist()

    print("Original array:", arr)

    # CPU timing
    arr_cpu = arr.copy()
    start = time.time()
    cpu_sorted = cpu_heap_sort(arr_cpu)
    cpu_time = time.time() - start
    print("CPU sorted:", cpu_sorted)
    print("CPU time:", cpu_time, "seconds")

    # GPU timing
    arr_gpu = arr.copy()
    start = time.time()
    gpu_result = gpu_heap_sort(arr_gpu)
    gpu_time = time.time() - start
    print("GPU heapify result (not fully sorted):", gpu_result.tolist())
    print("GPU time:", gpu_time, "seconds")
