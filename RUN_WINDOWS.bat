@echo off
rem Double-click to open the project menu (Windows, MSYS2 MinGW-w64 gcc).
rem Pass a command to skip the menu:  RUN_WINDOWS.bat all
cd /d "%~dp0"
where py >nul 2>nul && (py -3 run_project.py %* & goto :end)
where python >nul 2>nul && (python run_project.py %* & goto :end)
echo Python 3 is required -- see REQUIREMENTS.md
:end
pause
