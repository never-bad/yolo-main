param()
# 一键（幂等）启动平台：后端 8000 + 前端 3000，独立进程、日志落盘。
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