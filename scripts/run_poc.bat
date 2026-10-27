@echo off
REM Proof of Concept บน ICBHI: 6 วิธี x 3 seeds แล้วสรุปสถิติ
REM ใช้: ดับเบิลคลิกได้เลย (หา conda env lung ให้เอง)
cd /d "%~dp0.."
chcp 65001 > nul
set PYTHONIOENCODING=utf-8

REM ใช้ conda env "lung" ถ้ามี (ไม่ต้อง activate เอง) ไม่งั้นใช้ python ใน PATH
set "PY=%USERPROFILE%\anaconda3\envs\lung\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\miniconda3\envs\lung\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m src.prepare_data --dataset icbhi || goto :error
"%PY%" -m src.run_experiments --plan poc %* || goto :error
"%PY%" -m src.stats
"%PY%" -m src.stats --metric macro_f1
echo.
echo เสร็จแล้ว ดูผลใน results\
pause
exit /b 0

:error
echo มีขั้นตอนล้มเหลว ดูข้อความด้านบน (สั่งไฟล์นี้ซ้ำได้ จะข้ามรันที่เสร็จแล้ว)
pause
exit /b 1
