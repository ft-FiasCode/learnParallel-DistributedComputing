from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
data = rank

reduce_result = comm.reduce(data, op=MPI.SUM, root=0)
if rank == 0:
    print(f"Reduce sum = {reduce_result}")

allreduce_result = comm.allreduce(data, op=MPI.SUM)
print(f"Process {rank}: Allreduce sum = {allreduce_result}")


# Q3: how MPI_Allreduce differs from MPI_Reduce

# MPI_Reduce : Only the root process gets the result (recvbuf meaningful only on root)
#MPI_Allreduce : Every process in the communicator gets the same result (recvbuf filled identically on all processes)
