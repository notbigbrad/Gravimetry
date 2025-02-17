#!/bin/bash

# This will run in the parent directory
# To make this run in a different directory, uncomment the line below
# and replace "path" with the desired directory path
# cd "path"


echo "Running in $(pwd)"

# Open VS Code and Finder
open .

# Activate Python virtual environment
source .venv/bin/activate
python3 ./modules/testEnvironment.py

while true; do
    # Prompt for filename
    read -p "Enter python filename (or type 'exit' to quit): " filename
    if [[ "$filename" == "exit" ]]; then
        break
    fi

    # Run python file
    python3 "$filename".py
done

# Keep terminal open (optional)
exec bash
