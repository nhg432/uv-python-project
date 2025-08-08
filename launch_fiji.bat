@echo off
title Fiji - Cellular Imaging Analysis

echo.
echo ================================================
echo     Fiji - Cellular Imaging Analysis
echo ================================================
echo.
echo Fiji location: c:\Users\nhg43\OneDrive\Documents\code_directory\uv-python-project\fiji_new\fiji\fiji-windows-x64.exe
echo Ready for Zarr/N5 plugins
echo Perfect for OpenOrganelle data analysis
echo.

echo Starting Fiji...
echo (This may take a moment on first launch)
echo.

REM Launch Fiji
"c:\Users\nhg43\OneDrive\Documents\code_directory\uv-python-project\fiji_new\fiji\fiji-windows-x64.exe" %*

REM Handle exit codes
if errorlevel 1 (
    echo.
    echo Fiji encountered an error
    echo Try running as administrator if problems persist
    echo.
    pause
) else (
    echo.
    echo Fiji closed successfully
)
