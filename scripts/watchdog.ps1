param()
# 平台守护：每分钟由计划任务 YoloPlatformWatchdog 调用。
# 探测 8000 / 3000，端口不在则拉起；已在则跳过（幂等）。
# 日志：%TEMP%\yolo-watchdog.log
$Root = Split-Path -Parent $PSScriptRoot
$Log  = Join-Path $env:TEMP 'yolo-watchdog.log'
$ts   = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

try {
    $out = & (Join-Path $Root 'scripts\start_platform.ps1') 2>&1 | Out-String
    $line = "[$ts] $($out.Trim())"
    if ($out -match 'started') { $line += '  (ACTION TAKEN)' }
    Add-Content -Path $Log -Value $line
} catch {
    Add-Content -Path $Log -Value "[$ts] watchdog error: $($_.Exception.Message)"
}