from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  # Add this line

send_data = np.array([rank], dtype=np.int32)
gather_data = np.empty(size, dtype=np.int32)  # Use 'size' instead of comm.Get_size()

comm.Gather(send_data, gather_data, root=0)

if rank == 0:
    print(f"Gathered data at root: {gather_data}")
