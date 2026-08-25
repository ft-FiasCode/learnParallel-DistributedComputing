from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = ["Hello", "MPI", "from", "Rank0"]   

    req = comm.isend(data, dest=1)
    print("Rank 0 sent message, doing other work...")

    req.wait()
    print("Rank 0: Send completed.")

elif rank == 1:

    req = comm.irecv(source=0)
    print("Rank 1 waiting for message...")

    data = req.wait()

    print("Rank 1 received data:", data)