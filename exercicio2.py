import math
import random
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000

comm.Barrier()
inicio = time.time()

n_local = N // size + (1 if rank < N % size else 0)

rng = random.Random()
dentro = 0
for _ in range(n_local):
    x = rng.random()
    y = rng.random()
    if x * x + y * y <= 1.0:
        dentro += 1

total_dentro = comm.reduce(dentro, op=MPI.SUM, root=0)

if rank == 0:
    pi_estimado = 4.0 * total_dentro / N
    fim = time.time()

infos = comm.gather((rank, MPI.Get_processor_name(), n_local, dentro), root=0)

if rank == 0:
    print(f"N = {N} | processos = {size}")
    for r, host, n, d in infos:
        print(f"  rank {r} em {host}: {n} pontos, {d} dentro do circulo")
    print(f"Total de pontos dentro do circulo: {total_dentro}")
    print(f"PI aproximado: {pi_estimado:.6f} (erro absoluto: {abs(pi_estimado - math.pi):.6f})")
    print(f"Tempo distribuido MPI: {(fim - inicio) * 1000:.2f} ms")
