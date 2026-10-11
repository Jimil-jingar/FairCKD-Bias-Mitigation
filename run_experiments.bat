@echo off
REM FairCKD - re-run all experiments (about 45-60 minutes)
cd /d "%~dp0"
python -m pip install -r requirements.txt
python main.py --all
pause
