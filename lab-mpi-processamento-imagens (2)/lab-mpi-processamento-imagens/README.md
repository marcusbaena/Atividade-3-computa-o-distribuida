# Lab MPI - Processamento Distribuido de Imagens Medicas

Implementacao completa do laboratorio em Python, `mpi4py` e Docker Compose. O
cluster possui um `master` e tres `workers`, com dois slots por conteiner. O
programa usa explicitamente `Bcast`, `Scatter`, duas chamadas a `Barrier`,
`Reduce` e `Gather`.

## Estrutura

```text
processamento_imagens.py       programa MPI principal
gerar_resumo.py                consolida JSONs, CSV e grafico de speedup
Dockerfile                     imagem com Open MPI, SSH, Python e dependencias
docker-compose.yml             cluster master + 3 workers
hosts                          8 slots MPI distribuidos nos 4 conteineres
scripts/run_experiments.ps1    bateria completa no Windows/PowerShell
scripts/run_experiments.sh     bateria completa no Linux/macOS
relatorio/relatorio.tex        relatorio LaTeX pronto para preencher resultados
```

## 1. Pre-requisitos

- Docker Desktop iniciado, com `docker` e `docker compose` no terminal.
- Para compilar o relatorio localmente: uma distribuicao LaTeX com `latexmk`.
  Como alternativa, envie `relatorio.tex` e a pasta `figuras` ao Overleaf.

Nao e necessario instalar Python, MPI ou NumPy no computador: tudo executa nos
conteineres.

## 2. Subir e validar o cluster

Abra o PowerShell nesta pasta e execute:

```powershell
docker compose up -d --build
docker compose ps
```

Teste a comunicacao entre os quatro hosts:

```powershell
docker compose exec -T -u mpiuser master mpirun --hostfile /home/mpiuser/lab/hosts --bind-to none --map-by slot -np 8 hostname
```

A saida deve mencionar `master`, `worker1`, `worker2` e `worker3`.

## 3. Executar um teste rapido

O comando abaixo usa 4 processos e uma imagem 500 x 500:

```powershell
docker compose exec -T -u mpiuser master mpirun --hostfile /home/mpiuser/lab/hosts --bind-to none --map-by slot -np 4 python3 /home/mpiuser/lab/processamento_imagens.py --linhas 500 --colunas 500 --saida-json /home/mpiuser/lab/resultados/teste_rapido.json
```

O terminal exibira o relatorio consolidado, o diagnostico e a auditoria de cada
rank. O JSON aparecera na pasta `resultados` do computador.

## 4. Executar a bateria experimental

No PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run_experiments.ps1
```

O script executa tamanhos 2000, 4000 e 6000 com 1, 2, 4 e 8 processos, tres
vezes cada. O caso com 1 processo fornece a referencia para o speedup; as
capturas obrigatorias da atividade sao as execucoes com 2, 4 e 8 processos.

Para reduzir o tempo durante uma verificacao preliminar:

```powershell
.\scripts\run_experiments.ps1 -Tamanhos 2000 -Processos 2,4,8 -Repeticoes 1
```

Em Bash, use:

```bash
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

Ao final sao gerados:

- `resultados/resultado_N...json`: metricas estruturadas por execucao;
- `resultados/resultado_N...log`: saida que pode ser capturada para o relatorio;
- `resultados/resumo.csv`: medias, desvios, speedup e eficiencia;
- `relatorio/figuras/speedup.png`: grafico pronto para inclusao no LaTeX.

## 5. Capturas obrigatorias

Execute separadamente os comandos abaixo e capture o terminal completo. Troque
apenas o valor de `-np` entre 2, 4 e 8:

```powershell
docker compose exec -T -u mpiuser master mpirun --hostfile /home/mpiuser/lab/hosts --bind-to none --map-by slot -np 2 python3 /home/mpiuser/lab/processamento_imagens.py --linhas 2000 --colunas 2000
```

Salve as imagens como:

```text
relatorio/figuras/execucao_2_processos.png
relatorio/figuras/execucao_4_processos.png
relatorio/figuras/execucao_8_processos.png
relatorio/figuras/relatorio_consolidado.png
```

O arquivo LaTeX detecta automaticamente essas figuras. Enquanto elas nao
existirem, caixas reservadas aparecem no PDF.

## 6. Preencher e compilar o relatorio

1. Edite no inicio de `relatorio/relatorio.tex`: nome, RA, curso e data.
2. Copie os tempos medios de `resultados/resumo.csv` para as celulas marcadas
   como `PREENCHER`.
3. Confira se as quatro capturas e `speedup.png` estao em `relatorio/figuras`.
4. Compile:

```powershell
cd relatorio
latexmk -pdf -interaction=nonstopmode -halt-on-error relatorio.tex
```

Se usar Overleaf, envie `relatorio.tex` e a pasta `figuras`, selecione pdfLaTeX
e compile.

## 7. Encerrar o ambiente

```powershell
docker compose down
```

Esse comando remove os conteineres e a rede do laboratorio, mas preserva os
codigos, logs, JSONs, CSV e figuras no computador.

## Observacoes metodologicas

- A semente padrao (`2026`) garante a mesma imagem para comparar diferentes
  numeros de processos.
- A taxa sintetica padrao e 4% dos pixels pulmonares, produzindo normalmente um
  caso de atencao sem forcar artificialmente um quadro critico.
- `numpy.array_split` distribui corretamente o resto quando o numero de linhas
  nao e divisivel pelo numero de ranks; o caso 2003/4 pode ser testado com
  `--linhas 2003 --colunas 2003`.
- Ranks impares dormem `0.5 * rank` segundos depois da analise local, conforme o
  enunciado. Use `--sem-atraso` apenas para um experimento comparativo extra.
- A classificacao critica ocorre quando o percentual suspeito alcanca o limite
  critico ou quando o percentual altamente suspeito alcanca metade desse
  limite. Essa operacionalizacao torna objetiva a expressao "presenca
  expressiva" usada no enunciado e esta declarada no relatorio.
