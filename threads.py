import random
import sys
import threading
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
THREADS = 4

A = [[random.random() for _ in range(N)] for _ in range(N)]
B = [[random.random() for _ in range(N)] for _ in range(N)]
C = [[0] * N for _ in range(N)]


def calcular(inicio, fim):
    for i in range(inicio, fim):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]


inicio = time.time()
threads = []
linhas_por_thread = N // THREADS

for t in range(THREADS):
    ini = t * linhas_por_thread
    fim_t = N if t == THREADS - 1 else (t + 1) * linhas_por_thread
    th = threading.Thread(target=calcular, args=(ini, fim_t))
    threads.append(th)
    th.start()

for th in threads:
    th.join()

fim = time.time()

print(f"N = {N} | threads = {THREADS}")
print(f"C acumulado (soma dos elementos): {sum(map(sum, C)):.4f}")
print(f"Tempo com threads: {(fim - inicio) * 1000:.2f} ms")
