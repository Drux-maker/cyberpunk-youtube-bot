# Setup completo para Windows 11 + RTX 3060 (modo local sin APIs de pago)
# Ejecutar en PowerShell como administrador:
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\scripts\setup_windows_local.ps1

Write-Host "`n=== CyberpunkBot — Setup Windows Local ===" -ForegroundColor Cyan

# ─── 1. Verificar Python ──────────────────────────────────────────────────────
Write-Host "`n[1/7] Verificando Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python no encontrado. Descarga Python 3.12 de python.org" -ForegroundColor Red
    exit 1
}
Write-Host "OK: $pythonVersion" -ForegroundColor Green

# ─── 2. Verificar FFmpeg ──────────────────────────────────────────────────────
Write-Host "`n[2/7] Verificando FFmpeg..." -ForegroundColor Yellow
ffmpeg -version 2>&1 | Select-Object -First 1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FFmpeg no encontrado. Instalando via winget..." -ForegroundColor Yellow
    winget install Gyan.FFmpeg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Instala FFmpeg manualmente desde: https://ffmpeg.org/download.html" -ForegroundColor Red
        Write-Host "Añade el bin/ al PATH de Windows" -ForegroundColor Red
        exit 1
    }
}
Write-Host "OK: FFmpeg encontrado" -ForegroundColor Green

# ─── 3. Verificar CUDA ───────────────────────────────────────────────────────
Write-Host "`n[3/7] Verificando CUDA (RTX 3060)..." -ForegroundColor Yellow
nvidia-smi 2>&1 | Select-Object -First 3
if ($LASTEXITCODE -ne 0) {
    Write-Host "ADVERTENCIA: nvidia-smi no encontrado. Instala drivers NVIDIA actualizados." -ForegroundColor Yellow
}

# ─── 4. Entorno virtual ──────────────────────────────────────────────────────
Write-Host "`n[4/7] Creando entorno virtual..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
Write-Host "OK: venv activado" -ForegroundColor Green

# ─── 5. PyTorch con CUDA 12.1 ────────────────────────────────────────────────
Write-Host "`n[5/7] Instalando PyTorch CUDA 12.1 (RTX 3060)..." -ForegroundColor Yellow
Write-Host "Esto puede tardar varios minutos (~2.5GB de descarga)" -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar que CUDA funciona
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ─── 6. Dependencias base ────────────────────────────────────────────────────
Write-Host "`n[6/7] Instalando dependencias base..." -ForegroundColor Yellow
pip install -r requirements.txt

# ─── 7. Dependencias locales (MusicGen + SD) ─────────────────────────────────
Write-Host "`n[7/7] Instalando MusicGen + Stable Diffusion..." -ForegroundColor Yellow
Write-Host "Esto puede tardar 5-10 minutos" -ForegroundColor Gray

# xformers — reduce VRAM ~30%
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# HuggingFace Diffusers (Stable Diffusion)
pip install diffusers transformers accelerate safetensors

# AudioCraft (MusicGen de Meta)
pip install audiocraft

# ─── Descargar fuentes para thumbnails ───────────────────────────────────────
Write-Host "`nDescargando fuentes para thumbnails..." -ForegroundColor Yellow
python scripts/download_fonts.py

# ─── Setup .env ───────────────────────────────────────────────────────────────
if (-not (Test-Path ".env")) {
    Copy-Item "config/.env.example" ".env"
    Write-Host "`nCREADO: .env — edita este archivo y añade tu OPENAI_API_KEY" -ForegroundColor Yellow
    Write-Host "(Para el SEO de los títulos. Coste: ~0.01 EUR por vídeo)" -ForegroundColor Gray
}

Write-Host "`n=== SETUP COMPLETADO ===" -ForegroundColor Green
Write-Host ""
Write-Host "Siguiente paso — Test rápido (60 segundos de vídeo):" -ForegroundColor Cyan
Write-Host "  python scripts/test_local.py --style cyberpunk --duration 60 --no-seo" -ForegroundColor White
Write-Host ""
Write-Host "Test completo con SEO (necesita OPENAI_API_KEY en .env):" -ForegroundColor Cyan
Write-Host "  python scripts/test_local.py --style cyberpunk --duration 60" -ForegroundColor White
Write-Host ""
Write-Host "NOTA: La primera vez descarga los modelos (~5GB MusicGen + ~7GB SDXL)" -ForegroundColor Gray
Write-Host "Esto solo ocurre UNA vez. Las siguientes ejecuciones son inmediatas." -ForegroundColor Gray
