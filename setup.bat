@echo off
echo UPDATE PYTHON BEFORE CONTINUING (press enter when done)
pause >nul
@echo on

@REM call py -m venv .venv
@REM call .venv\Scripts\activate

call py -m pip install --upgrade pip

call py -m pip install numpy
call py -m pip install scipy
call py -m pip install matplotlib

call py -m pip install --upgrade numpy
call py -m pip install --upgrade scipy
call py -m pip install --upgrade matplotlib