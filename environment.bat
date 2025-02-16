@echo off

:: This will run in the parent directory
:: To make this run in a different directory, uncomment line below
:: and place the path of this folder in place of "path"
:: cd /d path

echo Running in %CD%

:: Open VsCode
call code .

:: Open into python virtual enviroment
call .venv\scripts\activate
py ./modules/testEnviroment.py

:loop
:: Prompt for filename
set /p filename=Enter python filename:

:: Run python file
py %filename%.py

:: Loop back
goto loop

:: Keep window open
cmd