from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm.Barrier()  
print(f"Process {rank} passed the barrier.")

# Q1: What is the role of MPI_Barrier in collective communication?

# MPI_Barrier(comm) is a synchronization operation.
# It blocks every process until all processes in that communicator have reached the same MPI_Barrier call, all processes wait become serialization
