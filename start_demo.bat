@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8000 in your browser. Keep this window open.
py review_server.py --port 8000
pause
