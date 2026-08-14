from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Master process (rank {rank}) of {size} processes")
else:
    print(f"Worker process (rank {rank}) reporting in")

comm.Barrier()  # Wait for all processes

print(f"Process {rank} finished.")
