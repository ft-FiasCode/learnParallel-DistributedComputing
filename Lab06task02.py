from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

feedback = {
    "region": rank,
    "rating": rank % 5 + 1,
    "comment": f"Feedback from region {rank}"
}

all_feedback = comm.allgather(feedback)

print(f"\nProcess {rank} received all feedback:")
for f in all_feedback:
    print(f)