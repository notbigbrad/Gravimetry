@echo off

:: This will run in the parent directory
:: To make this run in a different directory, uncomment line below
:: and place the path of this folder in place of "path"
:: cd /d path

echo Running in %CD%

:: Open VsCode and file explorer
call code .
start explorer .

:: Open into python virtual enviroment
@REM call .venv\scripts\activate
@REM py ./modules/testEnviroment.py

:loop
:: Prompt for filename
set /p filename=Enter python filename:
if /i "%filename%"=="exit" goto leave

:: Run python file
py %filename%.py

:: Loop back
goto loop
:leave
:: Keep window open
cmd
