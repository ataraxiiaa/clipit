# One-time setup: Python venv + dependencies + local models. Re-runnable.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Host "Installing uv (Python manager)..."; powershell -c "irm https://astral.sh/uv/install.ps1 | iex" }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Write-Host "Installing ffmpeg..."; winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { Write-Host "Installing Ollama..."; winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements }
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
Write-Host "Pulling the language model (about 1.9 GB, one time)..."
ollama pull qwen2.5:3b-instruct
Write-Host "Downloading the Whisper 'small' model (about 480 MB, one time)..."
.venv\Scripts\python.exe -c "from clipper import gpu; gpu.setup_cuda_dlls(); from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('whisper ready')"
Write-Host "Done. Start with start.bat"
