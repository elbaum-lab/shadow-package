@echo off
setlocal
cd /d "%~dp0"

call .venv\Scripts\activate.bat

python shadow_montage_tiltseries_gui_2d_3d.py --gui

if errorlevel 1 (
    echo.
    echo Processing ended with an error.
    pause
) else (
    echo.
    echo Processing finished.
    pause
)

endlocal