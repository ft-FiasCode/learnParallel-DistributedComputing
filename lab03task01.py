from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

work_time = rank + 1
print(f"Rank {rank}: start working .... (will take {work_time} seconds).", flush=True)
time.sleep(work_time)
print(f"Rank {rank}:  work complete", flush=True)

comm.Barrier()

if rank == 0:
    print("all processes have completed work", flush=True)

recv_req = comm.irecv(source=0, tag=77)

if rank == 0:
    msg = "THANK YOU"
    send_reqs = [comm.isend(msg, dest=i, tag=77) for i in range(size)]
else:
    send_reqs = []

received = recv_req.wait()
print(f"Rank {rank}: received message: {received}", flush=True)

for r in send_reqs:
    r.wait()
