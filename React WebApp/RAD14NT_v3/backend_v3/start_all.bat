@echo off
REM start_all.bat — Launch all model API servers in the cxr conda environment.
REM Place this file in your backend\ folder and double-click it, or run it from any terminal.
REM Each model opens in its own window. Close all windows or run stop_all.bat to stop.

set CONDA_ROOT=C:\Users\YJ\miniconda3
set ENV_NAME=cxr
set BACKEND=%~dp0

REM Path to conda's activation hook
set ACTIVATE=%CONDA_ROOT%\Scripts\activate.bat

REM Only launch models whose .pth file exists in checkpoints\
REM Add or remove lines below to match the files you actually have.

if exist "%BACKEND%checkpoints\resnet50_best_model.pth" (
    start "resnet50 :5001" cmd /k "call %ACTIVATE% %ENV_NAME% && cd /d %BACKEND% && python api.py --weights checkpoints\resnet50_best_model.pth --model resnet50 --port 5001"
) else (
    echo [skip] resnet50 — checkpoints\resnet50_best_model.pth not found
)

if exist "%BACKEND%checkpoints\efficientnet_best_model.pth" (
    start "efficientnet :5002" cmd /k "call %ACTIVATE% %ENV_NAME% && cd /d %BACKEND% && python api.py --weights checkpoints\efficientnet_best_model.pth --model efficientnet --port 5002"
) else (
    echo [skip] efficientnet — checkpoints\efficientnet_best_model.pth not found
)

if exist "%BACKEND%checkpoints\convnext_best_model.pth" (
    start "convnext :5003" cmd /k "call %ACTIVATE% %ENV_NAME% && cd /d %BACKEND% && python api.py --weights checkpoints\convnext_best_model.pth --model convnext --port 5003"
) else (
    echo [skip] convnext — checkpoints\convnext_best_model.pth not found
)

if exist "%BACKEND%checkpoints\swin_best_model.pth" (
    start "swin :5004" cmd /k "call %ACTIVATE% %ENV_NAME% && cd /d %BACKEND% && python api.py --weights checkpoints\swin_best_model.pth --model swin --port 5004"
) else (
    echo [skip] swin — checkpoints\swin_best_model.pth not found
)

if exist "%BACKEND%checkpoints\raddino_best_model.pth" (
    start "raddino :5005" cmd /k "call %ACTIVATE% %ENV_NAME% && cd /d %BACKEND% && python api.py --weights checkpoints\raddino_best_model.pth --model raddino --port 5005"
) else (
    echo [skip] raddino — checkpoints\raddino_best_model.pth not found
)

if exist "%BACKEND%checkpoints\radjepa_best_model.pth" (
    start "radjepa :5006" cmd /k "call %ACTIVATE% %ENV_NAME% && cd /d %BACKEND% && python api.py --weights checkpoints\radjepa_best_model.pth --model radjepa --port 5006"
) else (
    echo [skip] radjepa — checkpoints\radjepa_best_model.pth not found
)

echo.
echo All available servers launched. Minimise the windows and leave them running.
