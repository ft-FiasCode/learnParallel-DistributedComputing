from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()  


texts = [
    "hello world hello",
    "MPI is great for parallel computing", 
    "hello from MPI process",
    "parallel computing is powerful"
]
local_text = texts[rank % len(texts)]  
local_count = len(local_text.split())

print(f"Rank {rank}: '{local_text}' -- {local_count} words")

total_words = comm.reduce(local_count, op=MPI.SUM, root=0)

if rank == 0:
    print(f"\nTotal word count across all processes: {total_words}")
