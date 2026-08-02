from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# With barrier
comm.Barrier()
print(f"With barrier: Process {rank} done.")

# Without barrier
print(f"Without barrier: Process {rank} done.")

# Q4: What happens if you remove MPI_Barrier from the experiments? Compare the output.

# With MPI_Barrier → clean, ordered output; consistent timing; slower overall
# Without MPI_Barrier → messy/jumbled output; timing varies a lot; usually faster but harder to debug/verify
