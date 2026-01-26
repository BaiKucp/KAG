@echo off
echo Starting Medicine KAG API Server...
echo Fixing GBK encoding issue...
set PYTHONUTF8=1
cd /d "%~dp0"
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
pause
