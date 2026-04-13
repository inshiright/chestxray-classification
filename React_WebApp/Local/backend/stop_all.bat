@echo off
REM stop_all.bat — Kill all running api.py model servers.
echo Stopping all api.py processes...
taskkill /F /FI "WINDOWTITLE eq resnet50*"   2>nul
taskkill /F /FI "WINDOWTITLE eq efficientnet*" 2>nul
taskkill /F /FI "WINDOWTITLE eq convnext*"   2>nul
taskkill /F /FI "WINDOWTITLE eq swin*"       2>nul
taskkill /F /FI "WINDOWTITLE eq raddino*"    2>nul
taskkill /F /FI "WINDOWTITLE eq radjepa*"    2>nul
echo Done.
