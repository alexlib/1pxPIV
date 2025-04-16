#!/bin/bash
# Installation script for PIVSuite Python

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Create a new conda environment
echo "Creating a new conda environment 'pivsuite'..."
conda create -y -n pivsuite python=3.10

# Activate the environment
echo "Activating the environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pivsuite

# Install the package in development mode
echo "Installing PIVSuite in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -e ".[dev]"

echo "Installation complete!"
echo "To activate the environment, run: conda activate pivsuite"
echo "To run the tests, run: pytest"
echo "To use the CLI, run: pivsuite --help"
