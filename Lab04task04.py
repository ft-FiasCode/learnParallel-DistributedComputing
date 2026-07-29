from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
send_data = [10, 20, 30, 40] if rank == 0 else None
recv_data = comm.scatter(send_data, root=0)
doubled = recv_data * 2
results = comm.gather(doubled, root=0)
if rank == 0:
    print(f"Doubled values: {results}")