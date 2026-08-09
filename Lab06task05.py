from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    jobs = list(range(20))  
    num_workers = size - 1
    
    for i in range(1, size):
        if jobs:
            comm.send(jobs.pop(0), dest=i)
    
    while jobs:
        status = MPI.Status()
        finished_worker = comm.recv(source=MPI.ANY_SOURCE, status=status)
        worker_rank = status.Get_source()
        comm.send(jobs.pop(0), dest=worker_rank)

    
    for i in range(1, size):
        comm.send(None, dest=i)

else:
    while True:
        job = comm.recv(source=0)
        if job is None:
            break
        print(f"Worker {rank} processing job {job}")
        time.sleep(1)
        comm.send(rank, dest=0)