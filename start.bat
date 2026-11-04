@echo off
cd /d "%~dp0"
if not exist .venv (py -3.12 -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install -r requirements-web.txt
if not exist .env echo Внимание: .env не найден, API-анализ будет недоступен.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause