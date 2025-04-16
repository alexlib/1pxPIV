# Installation Guide for PIVSuite Python

This guide will help you install and set up the PIVSuite Python package.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Conda (recommended for managing environments)

## Installation Methods

### Method 1: Using Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pivsuite_py.git
   cd pivsuite_py
   ```

2. Run the installation script:
   ```bash
   ./install.sh
   ```

   This script will:
   - Create a new conda environment called `pivsuite`
   - Install the package in development mode
   - Install all required dependencies

3. Activate the environment:
   ```bash
   conda activate pivsuite
   ```

### Method 2: Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pivsuite_py.git
   cd pivsuite_py
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Install development dependencies (optional):
   ```bash
   pip install -e ".[dev]"
   ```

## Verifying the Installation

1. Run the tests:
   ```bash
   python -m pytest
   ```

2. Run a simple example:
   ```bash
   python examples/example_piv_analysis.py
   ```

## Using the Command-Line Interface

PIVSuite provides a command-line interface for common tasks:

```bash
# Show help
pivsuite --help

# Run PIV analysis on an image pair
pivsuite piv image1.tif image2.tif --output results

# Run BOS analysis on an image pair
pivsuite bos image1.tif image2.tif --output results
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure you have the latest version of pip and conda:
   ```bash
   pip install --upgrade pip
   conda update conda
   ```

2. Check that all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. If you're having issues with specific packages, try installing them separately:
   ```bash
   pip install numpy scipy matplotlib scikit-image
   ```

## Getting Help

If you need further assistance, please open an issue on the GitHub repository or contact the maintainers.
