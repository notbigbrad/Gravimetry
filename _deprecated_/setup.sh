#!/bin/bash

echo "UPDATE PYTHON BEFORE CONTINUING (press enter when done)"
read -r

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install matplotlib

# Upgrade dependencies
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade matplotlib
