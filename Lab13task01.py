import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def array_multiply_cpu(a, b, result):
    for i in prange(a.size):
        result[i] = a[i] * b[i]

# Using one-digit numbers
a = np.array([2, 5, 3, 8, 1, 9, 4, 7], dtype=np.float32)
b = np.array([3, 1, 6, 2, 8, 4, 5, 2], dtype=np.float32)
result = np.zeros_like(a)

array_multiply_cpu(a, b, result)

print("Array A :", a)
print("Array B :", b)
print("Result  :", result)

# NumPy validation
print("\nNumPy Result:", a * b)
print("Match:", np.allclose(result, a * b))