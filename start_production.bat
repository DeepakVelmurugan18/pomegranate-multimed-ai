@echo off
echo ===================================================================
echo   Starting Pomegranate-MultiMed AI Production Server
echo   Clinical White Theme | Multimodal | PDF Diagnostic Report
echo ===================================================================
echo.
cd /d "%~dp0"
set PORT=8080
set HOST=0.0.0.0

python app/server.py
pause
