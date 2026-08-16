from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

temp     = round(random.uniform(20, 40), 1)
humidity = round(random.uniform(30, 90), 1)
wind     = round(random.uniform(0, 50), 1)

print(f"Rank {rank}: temp={temp}°C, humidity={humidity}%, wind={wind}km/h")

sum_temp     = comm.reduce(temp,     op=MPI.SUM, root=0)
sum_humidity = comm.reduce(humidity, op=MPI.SUM, root=0)
sum_wind     = comm.reduce(wind,     op=MPI.SUM, root=0)

max_temp     = comm.reduce(temp,     op=MPI.MAX, root=0)
max_humidity = comm.reduce(humidity, op=MPI.MAX, root=0)
max_wind     = comm.reduce(wind,     op=MPI.MAX, root=0)

if rank == 0:
    print(f"\n--- Results from {size} sensors ---")
    print(f"Avg Temp:     {sum_temp/size:.1f}°C   | Max: {max_temp}°C")
    print(f"Avg Humidity: {sum_humidity/size:.1f}%  | Max: {max_humidity}%")
    print(f"Avg Wind:     {sum_wind/size:.1f}km/h  | Max: {max_wind}km/h")