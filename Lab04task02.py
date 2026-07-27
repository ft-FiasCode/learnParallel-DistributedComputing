from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_val = rank

sum_val = comm.reduce(local_val, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Sum of ranks = {sum_val}")


prid_val = comm.reduce(local_val, op=MPI.PROD, root=0)

if rank == 0:
    print(f"Product of ranks = {prid_val}")