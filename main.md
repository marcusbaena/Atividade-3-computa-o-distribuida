# Atividade Lab – Computação Distribuída com MPI: Multiplicação de Matrizes e Método de Monte Carlo

**Universidade Presbiteriana Mackenzie – Faculdade de Computação e Informática (FCI)**
**Disciplina:** Computação Distribuída
**Professores:** Prof. Alcides / Prof. Mário
**Aluno(a):** `[NOME COMPLETO]`
**RA:** `[RA]`
**Turma:** `[TURMA]`
**Data de entrega:** `[DD/MM/2026]`

---

## 1. Introdução e Fundamentação Teórica

O MPI (*Message Passing Interface*) é o padrão de programação paralela por troca de mensagens. Uma aplicação MPI é formada por vários processos independentes, cada um com seu próprio espaço de memória, que executam o mesmo programa e se diferenciam pelo seu **rank** (0 a *size* − 1). Em Python, o padrão é acessado pela biblioteca **mpi4py**.

**Comunicação coletiva.** Além da comunicação ponto a ponto (`send`/`recv`), o MPI oferece operações que envolvem todos os processos do comunicador:

- **`bcast` (broadcast):** o processo *root* envia o mesmo dado a todos os demais. Foi usado para distribuir as matrizes A e B.
- **`gather`:** cada processo envia um valor ao *root*, que recebe uma lista ordenada por rank. Foi usado para coletar as fatias da matriz C.
- **`reduce`:** combina os valores locais com uma operação (`MPI.SUM`, `MPI.MAX`, `MPI.MIN`, `MPI.PROD`) e entrega o resultado ao *root*. Foi usado para somar os pontos internos no Monte Carlo.

As funções com inicial minúscula (`bcast`, `gather`, `reduce`) trabalham com objetos Python quaisquer serializados via *pickle*, o que facilita o desenvolvimento, mas gera *overhead* de serialização. As versões com maiúscula (`Bcast`, `Gather`, ...) operam diretamente sobre buffers contíguos (ex.: `numpy.ndarray`).

**Divisão de tarefas (*data partitioning*).** O problema é dividido em partes independentes, cada uma processada por um processo. Na multiplicação de matrizes, cada linha `C[i]` depende apenas da linha `A[i]` e da matriz B. Por isso, as N linhas foram divididas em blocos contíguos de `N // size` linhas, e o último processo ficou também com o resto da divisão.

**Monte Carlo.** Sorteando pontos (x, y) uniformes em [0,1]², a probabilidade de um ponto cair no quarto de círculo x² + y² ≤ 1 é π/4. Logo, π ≈ 4 · (pontos dentro / total de pontos). Cada sorteio é independente, então o problema é "embaraçosamente paralelo": cada processo sorteia N/size pontos e só os contadores precisam ser comunicados.

---

## 2. Metodologia e Ambiente de Execução

### 2.1 Ambiente computacional

| Item | Valor |
|---|---|
| Host | `[GitHub Codespaces / Docker Desktop no Windows 11]` |
| CPU / RAM do host | `[ex.: 4 vCPUs, 16 GB – Codespaces]` |
| Docker / Docker Compose | `[saída de docker --version e docker compose version]` |
| Imagem base dos nós | Ubuntu 22.04 |
| Python | `[saída de python3 --version no master – esperado 3.10.x]` |
| MPI | OpenMPI `[saída de mpirun --version – esperado 4.1.x]` |
| mpi4py | `[versão exibida pelo script – esperado 3.1.x]` |

### 2.2 Topologia do cluster simulado

O cluster tem 4 containers Docker criados a partir da mesma imagem (`Dockerfile`) e ligados por uma rede *bridge* (`mpinet`) definida no `docker-compose.yml`:

```
                 rede bridge "mpinet"
   ┌──────────┬──────────────┬──────────────┬──────────────┐
   │  master  │   worker1    │   worker2    │   worker3    │
   │  rank 0  │   rank 1     │   rank 2     │   rank 3     │
   └──────────┴──────────────┴──────────────┴──────────────┘
      mpirun ──ssh──► inicia os processos nos workers
```

- O **master** executa o `mpirun`, que usa SSH sem senha (chave gerada na imagem e presente em `authorized_keys` de todos os nós) para iniciar os processos nos workers.
- O arquivo `hosts` declara 1 *slot* por nó, então `-np 4` coloca exatamente um processo em cada container.
- O `mpi4py` foi instalado pelo pacote `python3-mpi4py` do apt, compilado contra o mesmo OpenMPI do sistema.

### 2.3 Arquivos do projeto

| Arquivo | Descrição |
|---|---|
| `Dockerfile` | Imagem dos nós: OpenMPI, Python 3, mpi4py, SSH e usuário `mpiuser` |
| `docker-compose.yml` | Define os serviços `master`, `worker1`, `worker2`, `worker3` |
| `hosts` | Hostfile do OpenMPI |
| `sequencial.py` | Multiplicação de matrizes sequencial (código de referência) |
| `threads.py` | Multiplicação de matrizes com 4 threads (código de referência) |
| `exercicio1.py` | Multiplicação de matrizes distribuída (bcast + gather) |
| `pi_sequencial.py` | Monte Carlo sequencial (código de referência) |
| `exercicio2.py` | Monte Carlo distribuído (reduce com MPI.SUM) |

Os scripts recebem N pela linha de comando, por exemplo `python3 sequencial.py 600`. As versões sequencial e com threads mantêm exatamente o algoritmo do enunciado. Todos os programas foram executados **dentro do container master**, para que as três versões fossem comparadas no mesmo ambiente.

### 2.4 Implementação do Exercício 1

1. O rank 0 gera A e B (N × N) com `random.random()` e marca o tempo inicial.
2. `A = comm.bcast(A, root=0)` e `B = comm.bcast(B, root=0)` enviam as matrizes a todos os processos.
3. Cada rank calcula as linhas `[rank·N//size, (rank+1)·N//size)`, e o último rank vai até N.
4. `comm.gather((info, C_local), root=0)` devolve as fatias ao rank 0, já na ordem dos ranks.
5. O rank 0 concatena as fatias em C, imprime o tempo total (distribuição + cálculo + coleta), a soma dos elementos de C ("C acumulado") e o tempo de cálculo de cada rank/host.

### 2.5 Implementação do Exercício 2

1. `comm.Barrier()` sincroniza o início e o tempo começa a ser medido.
2. Cada processo sorteia `N // size` pontos com seu próprio gerador `random.Random()`, que recebe uma semente diferente do sistema operacional e evita sequências iguais entre processos.
3. `comm.reduce(dentro, op=MPI.SUM, root=0)` soma os contadores no rank 0.
4. O rank 0 calcula `π ≈ 4 · total_dentro / N` e imprime a estimativa, o erro absoluto e o tempo.

---

## 3. Resultados e Evidências de Execução

### 3.1 Subida do cluster

`[PRINT 1 – docker compose up -d --build e docker compose ps mostrando os 4 containers "running"]`

`[PRINT 2 – service ssh start nos 4 nós e conferência do mpi4py]`

### 3.2 Exercício 1 – Multiplicação de matrizes

`[PRINT 3 – execução de mpirun --hostfile hosts -np 4 python3 exercicio1.py 300 (e 600, 1000)]`

`[PRINT 4 – execução das versões sequencial e com threads]`

| Dimensão (N) | Tempo Sequencial (ms) | Tempo Multithreaded (ms) | Tempo Distribuído MPI (ms) |
|---|---|---|---|
| N = 300  | `[ ]` | `[ ]` | `[ ]` |
| N = 600  | `[ ]` | `[ ]` | `[ ]` |
| N = 1000 | `[ ]` | `[ ]` | `[ ]` |

**Speedup** (Tempo sequencial ÷ Tempo da versão):

| N | Speedup Threads | Speedup MPI | Eficiência MPI (Speedup/4) |
|---|---|---|---|
| 300  | `[ ]` | `[ ]` | `[ ]` |
| 600  | `[ ]` | `[ ]` | `[ ]` |
| 1000 | `[ ]` | `[ ]` | `[ ]` |

### 3.3 Exercício 2 – Monte Carlo (N = 10.000.000)

`[PRINT 5 – execução de mpirun --hostfile hosts -np 4 python3 exercicio2.py 10000000]`

`[PRINT 6 – execução de pi_sequencial.py]`

| Versão | π estimado | Erro absoluto | Tempo (ms) | Speedup |
|---|---|---|---|---|
| Sequencial | `[ ]` | `[ ]` | `[ ]` | 1,00 |
| Distribuída MPI (4 processos) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

---

## 4. Análise de Desempenho e Discussão

**Sequencial × Multithreaded.** No CPython, o *Global Interpreter Lock* (GIL) permite que só uma thread execute bytecode Python por vez. Como a multiplicação de matrizes em Python puro é 100% limitada por CPU, as 4 threads se alternam em vez de executar em paralelo. Assim, o tempo com threads fica próximo do sequencial ou até maior, por causa das trocas de contexto. `[Confirme: nos testes, a versão com threads levou X ms contra Y ms da sequencial.]`

**Sequencial × MPI.** Com MPI, cada rank é um processo separado com seu próprio interpretador e GIL, então há paralelismo real. O ganho teórico máximo com 4 processos é 4×, mas o speedup observado é menor por dois motivos:

1. **Overhead de comunicação e serialização:** `bcast` envia A e B (listas de listas de `float`) serializadas com *pickle* a cada processo, e `gather` traz as fatias de C de volta, também serializadas. Com N = 1000 são 2 milhões de floats em cada direção, e o custo de serializar e desserializar é de ordem O(N²).
2. **Recursos do host:** os 4 containers compartilham a mesma máquina física. Se o host tiver menos núcleos livres que 4, os processos disputam CPU.

Como o cálculo é O(N³) e a comunicação é O(N²), a fração de tempo gasta em comunicação diminui quando N cresce. Por isso, espera-se que o speedup do MPI **aumente com N**: para N = 300 o *overhead* pesa mais, e para N = 1000 o speedup se aproxima mais do ideal. `[Confirme com a tabela.]`

**Monte Carlo.** Aqui a comunicação é mínima: cada processo envia só um inteiro no `reduce`. Por isso, o speedup tende a ficar próximo do número de processos `[confirme]`. A precisão da estimativa depende apenas do total de pontos (erro ~ 1/√N), e não do número de processos. Com N = 10⁷, o erro esperado é da ordem de 10⁻³ a 10⁻⁴.

**Quando usar um cluster distribuído em vez de threads.**
- Threads compartilham memória e não têm custo de envio de dados. São adequadas para tarefas de E/S ou em linguagens/bibliotecas que liberam o GIL (ex.: NumPy).
- O cluster MPI se justifica quando a carga é pesada em CPU e grande o suficiente para que o ganho do paralelismo supere o custo de comunicação, ou quando o problema não cabe na memória ou nos núcleos de uma única máquina.
- Problemas com pouca comunicação, como Monte Carlo, escalam muito bem. Problemas que precisam transferir muitos dados, como a multiplicação de matrizes, só compensam para N grande. Nesse caso, também é recomendável usar buffers NumPy com `Bcast`/`Gatherv` para eliminar o *pickle*.

---

## 5. Dificuldades Encontradas e Soluções

| Dificuldade | Solução adotada |
|---|---|
| `[ex.: Docker não estava instalado no Windows local]` | `[ex.: execução no GitHub Codespaces, que já tem Docker]` |
| `[ex.: pip install mpi4py falhando por "externally-managed-environment" ou falta de compilador]` | `[ex.: uso do pacote python3-mpi4py do apt na imagem]` |
| `[ex.: mpirun pedindo confirmação de host key / senha SSH]` | `[ex.: chave SSH sem senha + StrictHostKeyChecking no no Dockerfile]` |
| `[ex.: aviso/erro de memória compartilhada (vader/CMA) no OpenMPI em container]` | `[ex.: btl_vader_single_copy_mechanism = none]` |
| `[ex.: tempo longo da versão sequencial para N = 1000]` | `[ex.: execução de um N por vez, aguardando a conclusão de cada teste]` |

---

## 6. Uso de Inteligência Artificial

**Ferramenta utilizada:** Claude (Anthropic), via extensão Claude Code no VS Code.

**Prompt principal:** envio do PDF do enunciado da atividade com a mensagem "preciso fazer essa atividade pode me ajudar". `[Acrescente outros prompts que você usar.]`

**Etapas em que a IA auxiliou:**
- Criação do `Dockerfile`, do `docker-compose.yml` e do `hosts` para o cluster master + 3 workers.
- Implementação de `exercicio1.py` e `exercicio2.py` e adaptação dos códigos de referência para receber N por argumento.
- Estrutura deste relatório e textos de fundamentação teórica e análise.

**Análise crítica:** `[Escreva com suas palavras. Sugestões do que avaliar: o código funcionou de primeira no cluster? Houve algum erro que você precisou corrigir? As previsões do texto de análise (GIL, speedup crescendo com N) se confirmaram com seus números? O que você entendeu por conta própria sobre bcast/gather/reduce ao testar?]`

---

## 7. Conclusão

`[Resuma com seus resultados. Ex.: A atividade mostrou na prática que threads em Python puro não aceleram tarefas limitadas por CPU por causa do GIL, enquanto o MPI obteve speedup de X× na multiplicação de matrizes (N = 1000) e Y× no Monte Carlo. Também ficou claro o impacto do custo de comunicação: problemas que trocam muitos dados só se beneficiam da distribuição para entradas grandes.]`

## Referências

- DALCIN, L. *MPI for Python (mpi4py) – Documentation*. Disponível em: https://mpi4py.readthedocs.io/
- MPI FORUM. *MPI: A Message-Passing Interface Standard, Version 4.1*. Disponível em: https://www.mpi-forum.org/docs/
- OPEN MPI PROJECT. *Open MPI Documentation*. Disponível em: https://docs.open-mpi.org/
- DOCKER INC. *Docker Compose Documentation*. Disponível em: https://docs.docker.com/compose/
- Material didático da disciplina: *Atividade Lab – Computação Distribuída com MPI: Multiplicação de Matrizes e Método de Monte Carlo*. Prof. Alcides / Prof. Mário, FCI – Mackenzie, 2026.

---

## Apêndice – Código-fonte

`[Cole aqui o conteúdo de exercicio1.py e exercicio2.py (ou anexe os arquivos junto na submissão).]`
