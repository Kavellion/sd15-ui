@echo off
REM SD 1.5 UI installer for Windows.
REM Python is bundled in the python_embeded folder - nothing to install first,
REM no venv is created, and no Python on your system is touched or needed.
REM The SD 1.5 model weights download on the first Generate, about 4 GB,
REM into the Hugging Face cache on your user profile - not into this folder.
setlocal
cd /d "%~dp0"

if exist python_embeded\python.exe goto HavePython
echo.
echo The bundled Python in python_embeded is missing.
echo Fully extract this package, then run install.bat again.
echo.
pause
exit /b 1

:HavePython
echo Bundled Python:
python_embeded\python.exe --version

if exist python_embeded\Scripts\pip.exe goto HavePip
echo Setting up pip for the bundled Python - first run only...
python_embeded\python.exe python_embeded\get-pip.py --no-warn-script-location
if not errorlevel 1 goto HavePip
echo.
echo Pip setup failed. Check your internet connection, then run this again.
echo.
pause
exit /b 1

:HavePip
python_embeded\python.exe -m pip install --no-warn-script-location --upgrade pip

echo Installing PyTorch with CUDA support - this is the big download...
python_embeded\python.exe -m pip install --no-warn-script-location torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto TorchOk
echo.
echo PyTorch install failed. Check your internet connection and that
echo your GPU driver is up to date, then run this again.
echo.
pause
exit /b 1

:TorchOk
echo Installing the remaining packages...
python_embeded\python.exe -m pip install --no-warn-script-location -r requirements.txt
if not errorlevel 1 goto ReqOk
echo.
echo Package install failed. Run this again - pip resumes
echo downloads it already finished.
echo.
pause
exit /b 1

:ReqOk
if not exist images_out mkdir images_out
if not exist models mkdir models
if not exist loras mkdir loras
echo.
echo Done. Double-click run.bat to start SD 1.5 UI.
echo.
pause
