@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

if /I "%~1"=="--self-test" goto :self_test

set "VENV_DIR=.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    call :find_python
    if not defined PYTHON_BOOTSTRAP (
        echo [ERROR] Python 3.11, 3.12, or 3.13 was not found.
        echo Install Python from https://www.python.org/downloads/windows/
        echo During installation, enable "Add python.exe to PATH".
        pause
        exit /b 1
    )
    echo [INFO] Creating the Windows virtual environment...
    call :create_venv
    if errorlevel 1 goto :failed
)

"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The existing .venv does not use Python 3.11-3.13.
    echo Delete the .venv folder and run this file again with a supported Python installed.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import accelerate, diffusers, gradio, numpy, openpyxl, pandas, peft, PIL, safetensors, torch, transformers" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing project dependencies. This can take several minutes...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto :failed
    "%VENV_PYTHON%" -m pip install -e ".[all]"
    if errorlevel 1 goto :failed
)

if not defined DEEPSEEK_API_KEY (
    echo.
    echo [INFO] DEEPSEEK_API_KEY is not set for this terminal.
    echo The key is used only by this process and is not written to the project.
    set /p "DEEPSEEK_API_KEY=Paste the DeepSeek API key, or press Enter to start without it: "
)

set "SDL_PATH=%~1"
if not defined SDL_PATH set "SDL_PATH=reference_data\颜色信息\SDL2_0.txt"

if exist "%SDL_PATH%" (
    echo [OK] SDL color table: %SDL_PATH%
) else (
    echo [WARN] SDL color table was not found: %SDL_PATH%
    echo [WARN] The demo will still run in module 3/5 preview mode.
    echo [WARN] To enable module 4, copy the authorized table to the path above,
    echo        or drag the SDL2_0.txt file onto this BAT file.
)

echo.
echo [INFO] Starting http://127.0.0.1:7860/
echo [INFO] The first generation downloads the Stable Diffusion base model.
"%VENV_PYTHON%" -m modules.module_06_demo_evaluation.src.app --inbrowser --sdl-path "%SDL_PATH%"
if errorlevel 1 goto :failed
exit /b 0

:find_python
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_BOOTSTRAP=py -3.12"
    exit /b 0
)
py -3.13 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_BOOTSTRAP=py -3.13"
    exit /b 0
)
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_BOOTSTRAP=py -3.11"
    exit /b 0
)
python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_BOOTSTRAP=python"
exit /b 0

:create_venv
%PYTHON_BOOTSTRAP% -m venv "%VENV_DIR%"
exit /b %errorlevel%

:self_test
set "VENV_DIR=%TEMP%\lighting-effect-agent-self-test-%RANDOM%-%RANDOM%"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
call :find_python
if not defined PYTHON_BOOTSTRAP (
    echo [ERROR] Python 3.11, 3.12, or 3.13 was not found.
    exit /b 1
)
call :create_venv
if errorlevel 1 goto :self_test_failed
"%VENV_PYTHON%" -c "import sys; print(sys.version)"
if errorlevel 1 goto :self_test_failed
rmdir /s /q "%VENV_DIR%"
echo [OK] Windows BAT bootstrap self-test passed.
exit /b 0

:self_test_failed
if exist "%VENV_DIR%" rmdir /s /q "%VENV_DIR%"
echo [ERROR] Windows BAT bootstrap self-test failed.
exit /b 1

:failed
echo.
echo [ERROR] Setup or startup failed. Review the messages above.
pause
exit /b 1
