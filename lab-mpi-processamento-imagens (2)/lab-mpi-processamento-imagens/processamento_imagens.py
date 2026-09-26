#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processamento distribuido de imagem medica sintetica com MPI."
    )
    parser.add_argument("--linhas", type=int, default=2000)
    parser.add_argument("--colunas", type=int, default=2000)
    parser.add_argument("--limiar-suspeito", type=int, default=200)
    parser.add_argument("--limiar-alto", type=int, default=230)
    parser.add_argument("--percentual-critico", type=float, default=5.0)
    parser.add_argument(
        "--taxa-anomalia",
        type=float,
        default=4.0,
        help="Percentual dos pixels pulmonares que recebe alteracoes sinteticas.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--fator-atraso",
        type=float,
        default=0.5,
        help="Segundos multiplicados pelo rank impar para simular straggler.",
    )
    parser.add_argument(
        "--sem-atraso",
        action="store_true",
        help="Desativa a heterogeneidade artificial (util apenas para comparacao).",
    )
    parser.add_argument(
        "--saida-json",
        type=Path,
        default=None,
        help="Arquivo JSON opcional com metricas e relatorios por rank.",
    )
    args = parser.parse_args()

    if args.linhas <= 0 or args.colunas <= 1:
        parser.error("linhas deve ser > 0 e colunas deve ser > 1")
    if not 0 <= args.limiar_suspeito <= 255:
        parser.error("limiar-suspeito deve estar entre 0 e 255")
    if not args.limiar_suspeito <= args.limiar_alto <= 255:
        parser.error("limiar-alto deve ser >= limiar-suspeito e <= 255")
    if not 0 < args.percentual_critico <= 100:
        parser.error("percentual-critico deve estar em (0, 100]")
    if not 0 <= args.taxa_anomalia <= 100:
        parser.error("taxa-anomalia deve estar entre 0 e 100")
    if args.fator_atraso < 0:
        parser.error("fator-atraso nao pode ser negativo")
    return args


def gerar_radiografia(parametros: dict[str, Any]) -> np.ndarray:
    
    linhas = parametros["linhas"]
    colunas = parametros["colunas"]
    rng = np.random.default_rng(parametros["seed"])

    eixo_y, eixo_x = np.ogrid[-1.0:1.0:complex(linhas), -1.0:1.0:complex(colunas)]
    gradiente = 14.0 * (eixo_y + 1.0) + 5.0 * np.abs(eixo_x)
    imagem = rng.normal(55.0, 9.0, size=(linhas, colunas)) + gradiente

    pulmao_esquerdo = ((eixo_x + 0.30) / 0.30) ** 2 + ((eixo_y + 0.02) / 0.78) ** 2 <= 1
    pulmao_direito = ((eixo_x - 0.30) / 0.30) ** 2 + ((eixo_y + 0.02) / 0.78) ** 2 <= 1
    mascara_pulmoes = pulmao_esquerdo | pulmao_direito
    imagem[mascara_pulmoes] = rng.normal(112.0, 17.0, int(mascara_pulmoes.sum()))

    candidatos = np.flatnonzero(mascara_pulmoes)
    quantidade = int(round(candidatos.size * parametros["taxa_anomalia"] / 100.0))
    if quantidade:
        alterados = rng.choice(candidatos, size=quantidade, replace=False)
        valores = rng.integers(
            parametros["limiar_suspeito"], 256, size=quantidade, dtype=np.uint16
        )
        imagem.reshape(-1)[alterados] = valores

    return np.clip(imagem, 0, 255).astype(np.uint8)


def classificar(
    percentual_suspeito: float,
    percentual_alto: float,
    percentual_critico: float,
) -> str:
    
    if (
        percentual_suspeito >= percentual_critico
        or percentual_alto >= percentual_critico / 2.0
    ):
        return "CRITICA"
    if percentual_suspeito >= 1.0:
        return "ATENCAO"
    return "NORMAL"


def intervalo_de_linhas(total: int, processos: int, rank: int) -> tuple[int, int]:
    
    quociente, resto = divmod(total, processos)
    quantidade = quociente + (1 if rank < resto else 0)
    inicio = rank * quociente + min(rank, resto)
    return inicio, quantidade


def main() -> None:
    args = argumentos()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    inicio_total = MPI.Wtime()

    parametros = None
    imagem_completa = None
    blocos = None

    if rank == 0:
        parametros = {
            "linhas": args.linhas,
            "colunas": args.colunas,
            "limiar_suspeito": args.limiar_suspeito,
            "limiar_alto": args.limiar_alto,
            "percentual_critico": args.percentual_critico,
            "taxa_anomalia": args.taxa_anomalia,
            "seed": args.seed,
            "fator_atraso": 0.0 if args.sem_atraso else args.fator_atraso,
        }
        print("[root] Gerando radiografia sintetica...", flush=True)
        imagem_completa = gerar_radiografia(parametros)
        
        blocos = np.array_split(imagem_completa, size, axis=0)

    
    parametros = comm.bcast(parametros, root=0)

    
    comm.Barrier()

    
    bloco_local = comm.scatter(blocos, root=0)
    linha_inicio, linhas_locais = intervalo_de_linhas(
        parametros["linhas"], size, rank
    )
    linha_fim = linha_inicio + linhas_locais - 1

    inicio_local = MPI.Wtime()
    total_local = int(bloco_local.size)
    soma_local = int(np.sum(bloco_local, dtype=np.uint64))
    maximo_local = int(np.max(bloco_local)) if total_local else -1

    mascara_suspeita = bloco_local >= parametros["limiar_suspeito"]
    mascara_alta = bloco_local >= parametros["limiar_alto"]
    suspeitos_local = int(np.count_nonzero(mascara_suspeita))
    altamente_suspeitos_local = int(np.count_nonzero(mascara_alta))

    coluna_meio = parametros["colunas"] // 2
    suspeitos_esquerda = int(np.count_nonzero(mascara_suspeita[:, :coluna_meio]))
    suspeitos_direita = int(np.count_nonzero(mascara_suspeita[:, coluna_meio:]))

    percentual_local = 100.0 * suspeitos_local / total_local if total_local else 0.0
    percentual_alto_local = (
        100.0 * altamente_suspeitos_local / total_local if total_local else 0.0
    )
    classificacao_local = classificar(
        percentual_local,
        percentual_alto_local,
        parametros["percentual_critico"],
    )

    atraso = parametros["fator_atraso"] * rank if rank % 2 == 1 else 0.0
    if atraso:
        time.sleep(atraso)
    tempo_local_ms = (MPI.Wtime() - inicio_local) * 1000.0

    
    comm.Barrier()

    
    total_global = comm.reduce(total_local, op=MPI.SUM, root=0)
    soma_global = comm.reduce(soma_local, op=MPI.SUM, root=0)
    maximo_global = comm.reduce(maximo_local, op=MPI.MAX, root=0)
    suspeitos_global = comm.reduce(suspeitos_local, op=MPI.SUM, root=0)
    altamente_suspeitos_global = comm.reduce(
        altamente_suspeitos_local, op=MPI.SUM, root=0
    )
    esquerda_global = comm.reduce(suspeitos_esquerda, op=MPI.SUM, root=0)
    direita_global = comm.reduce(suspeitos_direita, op=MPI.SUM, root=0)

    relatorio_local = {
        "rank": rank,
        "linha_inicio": linha_inicio,
        "linha_fim": linha_fim if linhas_locais else None,
        "linhas": linhas_locais,
        "dimensao": [linhas_locais, parametros["colunas"]],
        "pixels": total_local,
        "media": soma_local / total_local if total_local else 0.0,
        "maximo": maximo_local,
        "suspeitos": suspeitos_local,
        "altamente_suspeitos": altamente_suspeitos_local,
        "percentual_suspeito": percentual_local,
        "classificacao": classificacao_local,
        "atraso_s": atraso,
        "tempo_local_ms": tempo_local_ms,
    }

    
    relatorios = comm.gather(relatorio_local, root=0)

    if rank != 0:
        return

    tempo_total_ms = (MPI.Wtime() - inicio_total) * 1000.0
    media_global = soma_global / total_global
    percentual_global = 100.0 * suspeitos_global / total_global
    percentual_alto_global = 100.0 * altamente_suspeitos_global / total_global
    classificacao_global = classificar(
        percentual_global,
        percentual_alto_global,
        parametros["percentual_critico"],
    )
    if esquerda_global > direita_global:
        lado_critico = "ESQUERDO"
    elif direita_global > esquerda_global:
        lado_critico = "DIREITO"
    else:
        lado_critico = "EQUILIBRADO"

    print("\n" + "=" * 72)
    print("RELATORIO CONSOLIDADO DE TRIAGEM DISTRIBUIDA")
    print("=" * 72)
    print(f"Dimensoes do exame       : {parametros['linhas']} x {parametros['colunas']}")
    print(f"Processos MPI            : {size}")
    print(
        "Limiares                 : "
        f"suspeito >= {parametros['limiar_suspeito']}; "
        f"alto >= {parametros['limiar_alto']}; "
        f"critico >= {parametros['percentual_critico']:.2f}%"
    )
    print(f"Pixels analisados        : {total_global}")
    print(f"Intensidade media/global : {media_global:.2f} / {maximo_global}")
    print(f"Pixels suspeitos         : {suspeitos_global} ({percentual_global:.4f}%)")
    print(
        "Altamente suspeitos      : "
        f"{altamente_suspeitos_global} ({percentual_alto_global:.4f}%)"
    )
    print(f"Suspeitos E/D            : {esquerda_global} / {direita_global}")
    print(f"Lado mais comprometido   : {lado_critico}")
    print(f"Classificacao geral      : {classificacao_global}")
    print(f"Tempo total              : {tempo_total_ms:.3f} ms")

    print("\nAuditoria por processo:")
    for item in relatorios:
        intervalo = (
            "vazio"
            if item["linha_fim"] is None
            else f"{item['linha_inicio']}..{item['linha_fim']}"
        )
        print(
            f"  rank {item['rank']:02d} | linhas {intervalo:>11} | "
            f"suspeitos {item['suspeitos']:>9} | max {item['maximo']:>3} | "
            f"{item['classificacao']:<7} | {item['tempo_local_ms']:.3f} ms"
        )
    print("=" * 72, flush=True)

    resultado = {
        "parametros": parametros,
        "processos": size,
        "metricas_globais": {
            "pixels": total_global,
            "soma_intensidades": soma_global,
            "media_intensidades": media_global,
            "maximo": maximo_global,
            "suspeitos": suspeitos_global,
            "altamente_suspeitos": altamente_suspeitos_global,
            "percentual_suspeito": percentual_global,
            "percentual_alto": percentual_alto_global,
            "suspeitos_esquerda": esquerda_global,
            "suspeitos_direita": direita_global,
            "lado_critico": lado_critico,
            "classificacao": classificacao_global,
            "tempo_total_ms": tempo_total_ms,
        },
        "relatorios_por_rank": relatorios,
    }
    if args.saida_json:
        args.saida_json.parent.mkdir(parents=True, exist_ok=True)
        args.saida_json.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Resultado JSON salvo em: {args.saida_json}", flush=True)


if __name__ == "__main__":
    main()
