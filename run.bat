@echo off
REM Start SD 1.5 UI. Run install.bat once first.
REM
REM Pictures land in images_out\ next to this file, one subfolder per
REM batch name, with a settings.txt alongside the frames.
REM
REM The app runs on the Python bundled in python_embeded\ -- no
REM venv, no system Python involved.

setlocal
cd /d "%~dp0"
if not exist python_embeded\python.exe (
    echo The bundled Python is missing. Re-extract the package,
    echo then run install.bat.
    pause
    exit /b 1
)
if not exist python_embeded\Lib\site-packages\gradio (
    echo The packages are not installed yet. Run install.bat first.
    pause
    exit /b 1
)
python_embeded\python.exe app.py
echo.
echo The app stopped. Double-click run.bat to start it again.
pause
