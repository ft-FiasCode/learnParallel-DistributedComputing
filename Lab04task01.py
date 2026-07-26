
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.array([100], dtype=np.int32)  
    req = comm.Isend(data, dest=1, tag=11)
    print("Process 0 sent data:", data[0])
    req.Wait()  

elif rank == 1:
    data = np.array([0], dtype=np.int32)
    req = comm.Irecv(data, source=0, tag=11)
    
    req.Wait()  
    
    data[0] += 5
    
    print(f"Process 1 received {data[0]-5}, added 5, final value: {data[0]}")
