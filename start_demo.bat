@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8000 in your browser. Keep this window open.
py -m vmax_vision.serve --directory retraining_out/steward_review --port 8000
pause
