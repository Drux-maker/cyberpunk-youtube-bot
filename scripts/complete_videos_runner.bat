@echo off
setlocal

set "PROJECT=C:\Users\danie\Documents\Claude\cyberpunk_youtube_bot"
set "FFMPEG_BIN=C:\Users\danie\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"

cd /d "%PROJECT%"
set "PATH=%FFMPEG_BIN%;%PATH%"
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

if not exist logs mkdir logs

echo. >> logs\complete_videos.log
echo ============================================================ >> logs\complete_videos.log
echo  START %date% %time% >> logs\complete_videos.log
echo ============================================================ >> logs\complete_videos.log

".venv\Scripts\python.exe" scripts\complete_videos.py 1>> logs\complete_videos.log 2>&1

echo ============================================================ >> logs\complete_videos.log
echo  END   %date% %time%  exit=%errorlevel% >> logs\complete_videos.log
echo ============================================================ >> logs\complete_videos.log

endlocal
