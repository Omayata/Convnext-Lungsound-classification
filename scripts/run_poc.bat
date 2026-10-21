@echo off
REM Proof of Concept บน ICBHI: 6 วิธี x 3 seeds แล้วสรุปสถิติ
REM ใช้: เปิด Anaconda Prompt, conda activate lung, แล้วรันไฟล์นี้จากโฟลเดอร์ใดก็ได้
cd /d "%~dp0.."
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

python -m src.prepare_data --dataset icbhi || goto :error
python -m src.run_experiments --plan poc %* || goto :error
python -m src.stats
python -m src.stats --metric macro_f1
echo.
echo เสร็จแล้ว ดูผลใน results\
pause
exit /b 0

:error
echo มีขั้นตอนล้มเหลว ดูข้อความด้านบน (สั่งไฟล์นี้ซ้ำได้ จะข้ามรันที่เสร็จแล้ว)
pause
exit /b 1
