from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Scatter data from root to all processes
send_data = [i for i in range(size)] if rank == 0 else None
recv_data = comm.scatter(send_data, root=0)
print(f"Process {rank} got {recv_data}")

# Double the received value
local_result = recv_data * 2

# Gather results back to root
gathered = comm.gather(local_result, root=0)
if rank == 0:
    print(f"Gathered doubled values: {gathered}")


# Q5: Can MPI_Scatter and MPI_Gather be used together in a program? Justify your answer by implementing a program.

# Yes, MPI_Scatter and MPI_Gather work great together. The root process splits up a big job into equal pieces and sends one piece to each process using Scatter. Every process works on its own piece, then they all send their results back to the root using Gather..
