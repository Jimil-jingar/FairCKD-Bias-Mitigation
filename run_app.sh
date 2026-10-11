#!/usr/bin/env bash
# FairCKD - start the web application (Linux / macOS)
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 -m streamlit run app/Home.py
