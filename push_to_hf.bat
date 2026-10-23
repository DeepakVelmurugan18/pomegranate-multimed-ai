@echo off
setlocal
echo ===================================================================
echo   Deploy Pomegranate-MultiMed AI to Hugging Face Spaces (Docker)
echo ===================================================================
echo.
echo Step 1: Ensure you created a new Space at: https://huggingface.co/new-space
echo         - Select Space SDK: Docker (Blank)
echo.

set /p HF_USERNAME="Enter your Hugging Face username: "
set /p SPACE_NAME="Enter your Space name [default: pomegranate-multimed-ai]: "
if "%SPACE_NAME%"=="" set SPACE_NAME=pomegranate-multimed-ai

set REMOTE_URL=https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%

echo.
echo Setting remote 'space' to: %REMOTE_URL%
git remote remove space 2>nul
git remote add space %REMOTE_URL%

echo.
echo Pushing codebase to Hugging Face Space...
echo (When prompted for password, enter your HF Access Token from https://huggingface.co/settings/tokens)
echo.

git push --force space master:main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===================================================================
    echo   SUCCESS! Deployment submitted to Hugging Face!
    echo   Your Space is building at:
    echo   https://huggingface.co/spaces/%HF_USERNAME%/%SPACE_NAME%
    echo ===================================================================
) else (
    echo.
    echo Deployment push encountered an issue. Please verify your token and space name.
)

pause
