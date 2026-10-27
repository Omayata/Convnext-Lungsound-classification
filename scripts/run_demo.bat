@echo off
REM Demo สำหรับให้อาจารย์ดู: ICBHI จริง, 18 วิธี (เสนอ + คู่เทียบ 6 + ablation 11), seed 0, 5 epoch แล้วสร้าง results\demo\demo_report.html
REM ต้องมี datasets\ICBHI_2017\ และ checkpoints\convnext_tiny_471mAP.pth ก่อน  ใส่ --epochs 3 ต่อท้ายได้ถ้าอยากเร็วขึ้น
REM ใช้: ดับเบิลคลิกได้เลย (หา conda env lung ให้เอง)
cd /d "%~dp0.."
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

REM ใช้ conda env "lung" ถ้ามี (ไม่ต้อง activate เอง) ไม่งั้นใช้ python ใน PATH
set "PY=%USERPROFILE%\anaconda3\envs\lung\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\miniconda3\envs\lung\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m src.prepare_data --dataset icbhi || goto :error
"%PY%" -m src.run_experiments --plan demo %* || goto :error
"%PY%" scripts\demo_report.py || goto :error
start "" "results\demo\demo_report.html"
pause
exit /b 0

:error
echo มีขั้นตอนล้มเหลว ดูข้อความด้านบน (สั่งไฟล์นี้ซ้ำได้ จะข้ามรันที่เสร็จแล้ว)
pause
exit /b 1
