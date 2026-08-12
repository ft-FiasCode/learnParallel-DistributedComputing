from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

value = random.randint(1, 10)
print(f"Rank {rank}: value = {value}")

min_val = comm.allreduce(value, op=MPI.MIN)

if rank == 0:
    print(f"\nMinimum value = {min_val}")