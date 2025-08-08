@echo off
title Fiji Test

echo Testing Fiji Installation...
echo.

echo Fiji executable: c:\Users\nhg43\OneDrive\Documents\code_directory\uv-python-project\fiji_new\fiji\fiji-windows-x64.exe
echo.

if exist "c:\Users\nhg43\OneDrive\Documents\code_directory\uv-python-project\fiji_new\fiji\fiji-windows-x64.exe" (
    echo SUCCESS: Fiji executable found!
    echo.
    echo Next steps:
    echo 1. Double-click launch_fiji.bat to start Fiji
    echo 2. Follow FIJI_SETUP_GUIDE.md to add Zarr support
    echo 3. Test with OpenOrganelle data
) else (
    echo ERROR: Fiji executable not found
)

echo.
pause
