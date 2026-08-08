from mpi4py import MPI
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

processing_time = random.uniform(0.5, 2.0)
if rank == 1:
    time.sleep(3)

print(f"Server {rank} processing time: {processing_time:.2f}")

total_time = 0.0
request = comm.Iallreduce(processing_time, total_time, op=MPI.SUM)
print(f"Server {rank} doing other work while waiting...")

request.Wait()

average_time = total_time / size

print(f"Server {rank} calculated average processing time: {average_time:.2f}")