import pyopencl as cl
import numpy as np
import time

# CPU Heap Sort
def cpu_heapify(arr, n, i):
    largest = i
    l, r = 2*i+1, 2*i+2
    if l < n and arr[l] > arr[largest]: largest = l
    if r < n and arr[r] > arr[largest]: largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        cpu_heapify(arr, n, largest)

def cpu_heap_sort(arr):
    n = len(arr)
    for i in range(n//2-1, -1, -1):
        cpu_heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        cpu_heapify(arr, i, 0)

#  GPU Heapify Kernel 
kernel_code = """
__kernel void heapify(__global int* arr, int n, __global int* indices) {
    int gid = get_global_id(0);
    int i = indices[gid];

    int largest = i;
    int l = 2*i + 1;
    int r = 2*i + 2;

    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        int tmp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = tmp;
    }
}
"""

def gpu_heap_sort(data, ctx, queue):
    n = len(data)
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
    program = cl.Program(ctx, kernel_code).build()

    # Build max heap
    for i in range(n//2 - 1, -1, -1):
        indices = np.array([i], dtype=np.int32)
        idx_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=indices)
        program.heapify(queue, (1,), None, buf, np.int32(n), idx_buf)

    # Extract elements one by one
    for i in range(n-1, 0, -1):
        # swap root with end
        arr = np.empty_like(data)
        cl.enqueue_copy(queue, arr, buf).wait()
        arr[0], arr[i] = arr[i], arr[0]
        buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr)

        # heapify reduced heap
        indices = np.array([0], dtype=np.int32)
        idx_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=indices)
        program.heapify(queue, (1,), None, buf, np.int32(i), idx_buf)

    result = np.empty_like(data)
    cl.enqueue_copy(queue, result, buf).wait()
    return result

#  Main 
def main():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    data = np.random.randint(0, 100, size=16).astype(np.int32)
    print("Original:", data)

    # CPU sort
    cpu_arr = data.copy()
    start = time.time()
    cpu_heap_sort(cpu_arr)
    cpu_time = time.time() - start
    print("CPU Sorted:", cpu_arr, "Time:", cpu_time)

    # GPU sort
    start = time.time()
    gpu_arr = gpu_heap_sort(data.copy(), ctx, queue)
    gpu_time = time.time() - start
    print("GPU Sorted:", gpu_arr, "Time:", gpu_time)

if __name__ == "__main__":
    main()
