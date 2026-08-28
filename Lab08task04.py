from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
MAX_LEN = 50

if rank == 0:
    logs = [
        "INFO: System started",
        "INFO: Processing data", 
        "WARNING: High memory",
        "INFO: Task complete"
    ]
    for log in logs:
        buf = np.zeros(MAX_LEN, dtype='i')
        for i, c in enumerate(log[:MAX_LEN]):
            buf[i] = ord(c)
        
        req = comm.Isend([buf, MPI.INT], dest=1)
        print(f"Rank 0 sent log: {log}")
        req.Wait()
        time.sleep(0.5)

elif rank == 1:
    for _ in range(4):
        buf = np.zeros(MAX_LEN, dtype='i')
        req = comm.Irecv([buf, MPI.INT], source=0)
        print("Rank 1: Waiting for log...")
        req.Wait()
        
        msg = ''
        for c in buf:
            if c != 0:
                msg += chr(int(c))
        
        print(f"[LOG RECEIVED]: {msg}")
        time.sleep(0.3)
