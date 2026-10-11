@echo off
REM FairCKD - start the web application (Windows). Double-click this file.
cd /d "%~dp0"
python -m pip install -r requirements.txt
python -m streamlit run app/Home.py
pause
