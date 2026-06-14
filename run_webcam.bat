@echo off
title Smart Helmet Detection - Live Webcam
cd /d "%~dp0"
echo.
echo  ============================================================
echo   Smart Helmet Detection System
echo   Starting LIVE Webcam Detection ...
echo   Press Q in the video window to quit.
echo  ============================================================
echo.
python src/detect.py
pause
