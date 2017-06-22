@echo off

set WORKSPACE_BASE=%~dp0
set WAF_TOOLS_PATH=%WORKSPACE_BASE%\tools\build
set PYTHONPATH=%WAF_TOOLS_PATH%;%PYTHONPATH%

set OPENCV_BASE_PATH=C:\workspaces\dependencies\opencv\build
set OPENCV_BIN_PATH=%OPENCV_BASE_PATH%\x64\vc14\bin

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

set PATH=%OPENCV_BIN_PATH%;%PATH%
