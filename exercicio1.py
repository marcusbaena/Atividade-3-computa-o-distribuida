import random
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300

if rank == 0:
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
    inicio = time.time()
else:
    A = None
    B = None

A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

linhas_por_proc = N // size
ini = rank * linhas_por_proc
fim = N if rank == size - 1 else (rank + 1) * linhas_por_proc

t_calc = time.time()
C_local = [[0] * N for _ in range(fim - ini)]
for i in range(ini, fim):
    for j in range(N):
        for k in range(N):
            C_local[i - ini][j] += A[i][k] * B[k][j]
t_calc = time.time() - t_calc

info = (rank, MPI.Get_processor_name(), ini, fim, t_calc * 1000)

partes = comm.gather((info, C_local), root=0)

if rank == 0:
    C = [linha for _, fatia in partes for linha in fatia]
    fim_total = time.time()

    print(f"N = {N} | processos = {size}")
    for r, host, a, b, ms in (p[0] for p in partes):
        print(f"  rank {r} em {host}: linhas [{a}, {b}) calculadas em {ms:.2f} ms")
    print(f"Dimensao de C: {len(C)} x {len(C[0])}")
    print(f"C acumulado (soma dos elementos): {sum(map(sum, C)):.4f}")
    print(f"Tempo distribuido MPI: {(fim_total - inicio) * 1000:.2f} ms")
