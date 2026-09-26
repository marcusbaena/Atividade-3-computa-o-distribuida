#!/usr/bin/env python3


from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PADRAO = re.compile(r"resultado_N(?P<n>\d+)_P(?P<p>\d+)_R(?P<r>\d+)\.json$")


def main() -> None:
    raiz = Path(__file__).resolve().parent
    pasta_resultados = raiz / "resultados"
    grupos: dict[tuple[int, int], list[float]] = defaultdict(list)

    for arquivo in sorted(pasta_resultados.glob("resultado_N*_P*_R*.json")):
        correspondencia = PADRAO.match(arquivo.name)
        if not correspondencia:
            continue
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        chave = (int(correspondencia["n"]), int(correspondencia["p"]))
        grupos[chave].append(float(dados["metricas_globais"]["tempo_total_ms"]))

    if not grupos:
        raise SystemExit("Nenhum JSON de experimento foi encontrado em resultados/.")

    linhas = []
    for (n, p), tempos in sorted(grupos.items()):
        linhas.append(
            {
                "N": n,
                "processos": p,
                "repeticoes": len(tempos),
                "tempo_medio_ms": statistics.fmean(tempos),
                "desvio_padrao_ms": statistics.stdev(tempos) if len(tempos) > 1 else 0.0,
            }
        )

    baselines: dict[int, tuple[int, float]] = {}
    for item in linhas:
        atual = baselines.get(item["N"])
        if atual is None or item["processos"] < atual[0]:
            baselines[item["N"]] = (item["processos"], item["tempo_medio_ms"])
    for item in linhas:
        p_base, tempo_base = baselines[item["N"]]
        item["processos_base"] = p_base
        item["speedup"] = tempo_base / item["tempo_medio_ms"]
        item["eficiencia"] = item["speedup"] / (item["processos"] / p_base)

    destino_csv = pasta_resultados / "resumo.csv"
    with destino_csv.open("w", newline="", encoding="utf-8") as saida:
        escritor = csv.DictWriter(saida, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    for n in sorted({item["N"] for item in linhas}):
        subconjunto = [item for item in linhas if item["N"] == n]
        plt.plot(
            [item["processos"] for item in subconjunto],
            [item["speedup"] for item in subconjunto],
            marker="o",
            label=f"{n} x {n}",
        )
    plt.xlabel("Numero de processos MPI")
    plt.ylabel("Speedup relativo ao menor P medido")
    plt.title("Speedup do processamento distribuido")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figura = raiz / "relatorio" / "figuras" / "speedup.png"
    figura.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figura, dpi=180)
    print(f"Resumo salvo em {destino_csv}")
    print(f"Grafico salvo em {figura}")


if __name__ == "__main__":
    main()
