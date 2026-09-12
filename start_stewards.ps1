$ErrorActionPreference = 'Stop'
$demoUrl = 'http://127.0.0.1:8010'
$demoReady = $false
try {
    $demoSession = Invoke-RestMethod "$demoUrl/api/clips" -TimeoutSec 2
    $demoReady = $null -ne $demoSession.clips
} catch {}
if (-not $demoReady) {
    $demoPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $demoLogs = Join-Path $env:USERPROFILE 'trackshift_runs'
    New-Item -ItemType Directory -Path $demoLogs -Force | Out-Null
    Start-Process -FilePath $demoPython -ArgumentList 'vmax_live_server.py' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoLogs 'vmax_demo_server.log') -RedirectStandardError (Join-Path $demoLogs 'vmax_demo_server.error.log')
    for ($demoAttempt = 0; $demoAttempt -lt 20; $demoAttempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $demoSession = Invoke-RestMethod "$demoUrl/api/clips" -TimeoutSec 1
            $demoReady = $null -ne $demoSession.clips
            if ($demoReady) { break }
        } catch {}
    }
}
if (-not $demoReady) { throw 'The demo server did not start. See trackshift_runs/vmax_demo_server.error.log.' }
Start-Process $demoUrl
