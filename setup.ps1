# One-command setup for Windows.  Run from the repo folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1          (auto GPU/CPU)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Cpu     (force CPU builds)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -SkipDeep
#
# Creates two separate environments (they need different PyTorch versions):
#   .venv                                         fast model, export, server, tests
#   vmax_model2/Track_limit_detection/.venv_bench deep model
param([switch]$Cpu, [switch]$SkipDeep)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Copy instead of hardlinking from uv's cache: hardlinks fail inside OneDrive
# and other cloud-synced folders ("incompatible hardlinks", os error 396).
$env:UV_LINK_MODE = 'copy'

function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "WARNING: $m" -ForegroundColor Yellow }
function Run { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE -ne 0) { throw "Command failed: $($args -join ' ')" } }

# --- uv (fast installer; also downloads Python 3.12 if you don't have it) ---
Step 'Checking for uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) { throw 'Neither uv nor python was found. Install uv from https://docs.astral.sh/uv/ and re-run.' }
    Write-Host 'uv not found; installing it with pip'
    Run python -m pip install --user --quiet uv
    $uvDir = (& python -c "import sysconfig,os;print(sysconfig.get_path('scripts', f'{os.name}_user'))").Trim()
    $env:PATH = "$uvDir;$env:PATH"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv installed but not on PATH. Add $uvDir to PATH and re-run." }
}
Run uv --version

# --- Git LFS: the deep model checkpoint ---
Step 'Checking the deep model checkpoint (Git LFS)'
$ckpt = 'vmax_model2/Track_limit_detection/training_runs/boundary_heatmaps_20260912_191710_183913/best_model/boundary_heatmaps.pt'
if ((Test-Path $ckpt) -and (Get-Item $ckpt).Length -lt 1MB) {
    if (Get-Command git-lfs -ErrorAction SilentlyContinue) { Run git lfs install --local; Run git lfs pull }
    else { Warn 'Git LFS is not installed, so the deep model checkpoint is only a placeholder. Install it from https://git-lfs.com, then run: git lfs pull' }
}
if ((Test-Path $ckpt) -and (Get-Item $ckpt).Length -ge 1MB) { Write-Host ('Checkpoint OK ({0:N0} MB)' -f ((Get-Item $ckpt).Length / 1MB)) }

# --- Pick PyTorch builds ---
Step 'Choosing PyTorch builds'
$driver = 0
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $raw = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($raw) { $driver = [double]($raw.Trim().Split('.')[0]) }
}
# torch 2.14 is published for CUDA 13.0 (driver >= 580) or CPU;
# torch 2.11 for CUDA 12.8 (driver >= 570) or CPU.
$fastIndex = if ($driver -ge 580) { 'cu130' } else { 'cpu' }
$deepIndex = if ($driver -ge 570) { 'cu128' } else { 'cpu' }
if ($driver -eq 0) { Write-Host 'No NVIDIA GPU detected (or -Cpu given): installing CPU builds. Everything works, inference is slower.' }
elseif ($driver -lt 580) { Warn "NVIDIA driver $driver is older than 580: the fast model uses the CPU build. Update the driver for GPU speed." }
Write-Host "Fast model: torch 2.14.0 ($fastIndex)   Deep model: torch 2.11.0 ($deepIndex)"

# --- Main environment ---
Step 'Creating .venv (fast model, server, tests)'
Run uv venv .venv --python 3.12 --allow-existing
$fastPy = '.venv/Scripts/python.exe'
Run uv pip install --python $fastPy torch==2.14.0 torchvision==0.29.0 --index-url "https://download.pytorch.org/whl/$fastIndex"
Run uv pip install --python $fastPy -r requirements-fast.txt
Run $fastPy -c "import torch, ultralytics, cv2, numpy; print('fast env OK: torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# --- Deep model environment ---
if (-not $SkipDeep) {
    Step 'Creating .venv_bench (deep model)'
    $deepDir = 'vmax_model2/Track_limit_detection'
    Run uv venv "$deepDir/.venv_bench" --python 3.12 --allow-existing
    $deepPy = "$deepDir/.venv_bench/Scripts/python.exe"
    Run uv pip install --python $deepPy torch==2.11.0 torchvision==0.26.0 --index-url "https://download.pytorch.org/whl/$deepIndex"
    Run uv pip install --python $deepPy -r "$deepDir/requirements-deep.txt"
    Run $deepPy -c "import torch, transformers; print('deep env OK: torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA available:', torch.cuda.is_available())"
}

# --- FFmpeg (only needed for the simulator/rendering, not the demo) ---
Step 'Checking FFmpeg'
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Write-Host 'FFmpeg found' }
elseif ($env:FFMPEG) { Write-Host "Using FFMPEG=$env:FFMPEG" }
else { Warn 'FFmpeg not found. Only needed to render new simulator clips: winget install Gyan.FFmpeg' }

Step 'Done'
Write-Host 'Start the demo:  .venv\Scripts\python.exe vmax_live_server.py   then open http://127.0.0.1:8010'
Write-Host 'Or double-click start_demo.bat'
