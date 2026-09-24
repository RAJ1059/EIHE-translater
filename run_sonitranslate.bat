@echo off
cd /d "%~dp0"
echo Starting SoniTranslate...
call .venv\Scripts\activate.bat
python app_rvc.py %*
pause

