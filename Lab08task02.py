from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {"hostname": socket.gethostname(), "rank": rank}
else:
    data = None

data = comm.bcast(data, root=0)

print(f"Rank {rank} received: {data}")