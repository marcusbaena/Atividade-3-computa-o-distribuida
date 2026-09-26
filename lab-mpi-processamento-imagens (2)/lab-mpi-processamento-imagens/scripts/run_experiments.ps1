param(
    [int[]]$Tamanhos = @(2000, 4000, 6000),
    [int[]]$Processos = @(1, 2, 4, 8),
    [int]$Repeticoes = 3
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz
New-Item -ItemType Directory -Force -Path "resultados" | Out-Null

docker compose up -d --build
if ($LASTEXITCODE -ne 0) { throw "Falha ao iniciar o cluster Docker." }

foreach ($n in $Tamanhos) {
    foreach ($p in $Processos) {
        foreach ($r in 1..$Repeticoes) {
            $base = "resultado_N${n}_P${p}_R${r}"
            Write-Host "Executando N=$n, P=$p, repeticao=$r..." -ForegroundColor Cyan
            $argumentosDocker = @(
                "compose", "exec", "-T", "-u", "mpiuser", "master",
                "mpirun", "--hostfile", "/home/mpiuser/lab/hosts",
                "--bind-to", "none", "--map-by", "slot", "-np", "$p",
                "python3", "/home/mpiuser/lab/processamento_imagens.py",
                "--linhas", "$n", "--colunas", "$n", "--seed", "2026",
                "--saida-json", "/home/mpiuser/lab/resultados/${base}.json"
            )
            $saida = & docker @argumentosDocker 2>&1
            $codigo = $LASTEXITCODE
            $saida | Tee-Object -FilePath "resultados/${base}.log"
            if ($codigo -ne 0) { throw "Falha em N=$n, P=$p, repeticao=$r." }
        }
    }
}

docker compose exec -T -u mpiuser master `
    python3 /home/mpiuser/lab/gerar_resumo.py
if ($LASTEXITCODE -ne 0) { throw "Falha ao consolidar os resultados." }

Write-Host "Experimentos concluidos. Consulte resultados/resumo.csv." -ForegroundColor Green
