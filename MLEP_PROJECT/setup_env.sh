#!/usr/bin/env bash
# Quick Environment Setup Script for MLEP Project
set -e

echo "================================================================================"
echo "setting up Python Environment & Downloading Required Packages"
echo "================================================================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "[INFO] Conda detected. You can create the conda environment by running:"
    echo "       conda env create -f environment.yml"
    echo "       conda activate venv"
    echo ""
fi

echo "[INFO] Creating standard Python virtual environment 'venv'..."
python3 -m venv venv

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Upgrading pip and wheel..."
pip install --upgrade pip wheel setuptools

echo "[INFO] Installing required project dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "================================================================================"
echo "SUCCESS! Environment created and all packages downloaded & installed."
echo "To activate your environment in terminal, run:"
echo "    source venv/bin/activate"
echo "To verify the project installation and run unit tests, execute:"
echo "    pytest tests/ -v"
echo "================================================================================"
