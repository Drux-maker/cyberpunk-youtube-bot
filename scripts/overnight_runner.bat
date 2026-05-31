@echo off
REM ==============================================================
REM Overnight runner — independiente de la sesion de Claude Code.
REM Se ejecuta via Windows Task Scheduler para que NO muera al
REM cerrar terminales ni sesiones. Logs van a logs\overnight_run.log.
REM ==============================================================
setlocal

set "PROJECT=C:\Users\danie\Documents\Claude\cyberpunk_youtube_bot"
set "FFMPEG_BIN=C:\Users\danie\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"

cd /d "%PROJECT%"

set "PATH=%FFMPEG_BIN%;%PATH%"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

if not exist logs mkdir logs

REM Marca de inicio en el log para auditoria
echo. >> logs\overnight_run.log
echo ============================================================ >> logs\overnight_run.log
echo  START %date% %time%  PID=%RANDOM% >> logs\overnight_run.log
echo ============================================================ >> logs\overnight_run.log

REM Ejecutar y volcar stdout+stderr al log
".venv\Scripts\python.exe" scripts\overnight_av.py 1>> logs\overnight_run.log 2>&1

echo ============================================================ >> logs\overnight_run.log
echo  END   %date% %time%  exit=%errorlevel% >> logs\overnight_run.log
echo ============================================================ >> logs\overnight_run.log

endlocal
