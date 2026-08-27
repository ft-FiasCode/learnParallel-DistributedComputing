from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

value = np.array([rank + 1], dtype='f')
result = np.zeros(1, dtype='f')

req = comm.Iallreduce(value, result, op=MPI.SUM)
print("Rank", rank, "doing other work...")
req.Wait()

average = result[0] / size

print("Rank", rank, "-> sum =", result[0], ", average =", average)