param()
# 一键（幂等）启动平台：元数据库 5432 + 后端 8000 + 前端 3000，独立进程、日志落盘。
# 端口已在监听则跳过，可安全重复执行 / 被守护任务反复调用。
$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$Py   = 'D:\miniconda\envs\yolo-main\python.exe'
$Backend = Join-Path $Root 'backend'
$Front   = Join-Path $Root 'frontend'
$DataDir = Join-Path $Backend 'data'
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

function Test-PortListening([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Start-Detached([string]$File, [string[]]$ArgList, [string]$WorkDir, [string]$Out, [string]$Err) {
    # -WindowStyle Hidden：避免守护拉服务时弹出控制台窗口（弹窗被误关=误杀服务）
    $p = Start-Process -FilePath $File -ArgumentList $ArgList -WorkingDirectory $WorkDir `
        -RedirectStandardOutput $Out -RedirectStandardError $Err -WindowStyle Hidden -PassThru
    return $p.Id
}

$report = @()

# ---- 元数据库 PostgreSQL（便携版随仓库解压在上层目录；算法库 model_libraries 等表只存在 PG 里）----
# 必须在后端之前拉起：后端启动时的 init_db() 连不上 PG 会静默跳过，之后访问算法库接口就报 500。
$RepoRoot   = Split-Path -Parent $Root
$PgCtl      = @((Join-Path $RepoRoot 'pgsql\bin\pg_ctl.exe'), (Join-Path $Root 'pgsql\bin\pg_ctl.exe')) |
              Where-Object { Test-Path $_ } | Select-Object -First 1
$PgData     = @((Join-Path $RepoRoot 'pgdata'), (Join-Path $Root 'pgdata')) |
              Where-Object { Test-Path (Join-Path $_ 'PG_VERSION') } | Select-Object -First 1
$PgLog      = Join-Path $DataDir 'pg.log'

if (Test-PortListening 5432) {
    $report += 'postgres already running'
} elseif (-not $PgCtl -or -not $PgData) {
    $report += 'postgres SKIP: pg_ctl.exe / pgdata not found (library API will return 500)'
} else {
    & $PgCtl -D $PgData -l $PgLog -t 30 start | Out-Null
    # pg_ctl 返回后端口未必立即可连，轮询最多 ~10s
    $waited = 0
    while (-not (Test-PortListening 5432) -and $waited -lt 10) { Start-Sleep -Seconds 1; $waited++ }
    if (Test-PortListening 5432) { $report += "postgres started ($PgData)" }
    else { $report += "postgres FAILED to start, see $PgLog" }
}

if (-not (Test-PortListening 8000)) {
    $id = Start-Detached $Py @('run.py') $Backend "$DataDir\server.log" "$DataDir\server.err.log"
    $report += "backend started (pid $id)"
} else {
    $report += 'backend already running'
}
if (-not (Test-PortListening 3000)) {
    $id = Start-Detached 'npm.cmd' @('run','dev') $Front "$Front\vite.log" "$Front\vite.err.log"
    $report += "frontend started (pid $id)"
} else {
    $report += 'frontend already running'
}
$report | ForEach-Object { Write-Output $_ }
# 数字足够大时脚本输出会被截断，这里确保退出码正常
exit 0