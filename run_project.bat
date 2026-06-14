@echo off
title Smart Helmet Detection - Dashboard
cd /d "%~dp0"
echo.
echo  ============================================================
echo   Smart Helmet Detection System
echo   Starting Streamlit Dashboard ...
echo  ============================================================
echo.
python -m streamlit run src/dashboard.py --server.headless false
pause
