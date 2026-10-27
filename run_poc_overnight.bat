@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set "PY=C:\Users\tonwa\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"
set "LOG=results\poc_overnight_log.txt"
echo ============================================================
echo  PoC run: proposed + abl_mel_only, seed 0, max 50 epochs
echo  Log file: %CD%\%LOG%
echo  Do not close this window. Keep the laptop plugged in and awake.
echo ============================================================
echo [%date% %time%] START >> "%LOG%"
"%PY%" -u -m src.run_experiments --plan poc --methods proposed abl_mel_only --seeds 0 --save_ckpt >> "%LOG%" 2>&1
echo [%date% %time%] EXIT CODE %errorlevel% >> "%LOG%"
echo.
echo Finished with exit code %errorlevel%. See %LOG%
pause
