#!/bin/bash

echo "===================================="
echo "CV Validator Application Starting..."
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python එක install කරලා නෑ!"
    echo "කරුණාකර Python install කරන්න"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import flask" &> /dev/null
then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Starting CV Validator..."
echo "Open your browser and go to: http://127.0.0.1:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 cv_validator_app.py
