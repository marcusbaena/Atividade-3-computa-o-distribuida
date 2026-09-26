#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p resultados

REPETICOES="${REPETICOES:-3}"
docker compose up -d --build

for n in 2000 4000 6000; do
  for p in 1 2 4 8; do
    for r in $(seq 1 "$REPETICOES"); do
      base="resultado_N${n}_P${p}_R${r}"
      echo "Executando N=$n, P=$p, repeticao=$r..."
      docker compose exec -T -u mpiuser master \
        mpirun --hostfile /home/mpiuser/lab/hosts \
        --bind-to none --map-by slot -np "$p" \
        python3 /home/mpiuser/lab/processamento_imagens.py \
        --linhas "$n" --colunas "$n" --seed 2026 \
        --saida-json "/home/mpiuser/lab/resultados/${base}.json" \
        2>&1 | tee "resultados/${base}.log"
    done
  done
done

docker compose exec -T -u mpiuser master \
  python3 /home/mpiuser/lab/gerar_resumo.py

echo "Experimentos concluidos. Consulte resultados/resumo.csv."
