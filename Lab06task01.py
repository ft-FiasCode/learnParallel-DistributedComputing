from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    monthly_sales = [12000, 15000, 18000, 20000,
                     17000, 22000, 25000, 24000,
                     21000, 23000, 26000, 30000]
    
    chunk_size = len(monthly_sales) // size
    data_chunks = [monthly_sales[i*chunk_size:(i+1)*chunk_size] 
                   for i in range(size)]
else:
    data_chunks = None

local_sales = comm.scatter(data_chunks, root=0)

print(f"Process {rank} received monthly sales: {local_sales}")