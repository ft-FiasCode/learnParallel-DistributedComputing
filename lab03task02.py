from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    message = 'huzaifa'
    sendstr = comm.isend(message, dest=1, tag=1)
    sendstr.wait()
    print(f'process 0 --> sent: {message}')

elif rank == 1:
    recvstr = comm.irecv(source=0, tag=1 )
    message = recvstr.wait()
    print(f'process 1 <-- received: {message}')
