from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

local_revenue = 10000 + rank * 5000

all_revenues = comm.gather(local_revenue, root=0)

if rank == 0:
    median_value = np.median(all_revenues)
    print("All revenues:", all_revenues)
    print("Median revenue:", median_value)