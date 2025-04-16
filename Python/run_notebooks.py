#!/usr/bin/env python3
"""
Script to run Jupyter notebooks.

This script starts a Jupyter notebook server in the notebooks/ directory.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start a Jupyter notebook server."""
    # Get the notebooks directory
    notebooks_dir = Path(__file__).parent / "notebooks"
    
    # Check if the directory exists
    if not notebooks_dir.exists():
        print(f"Error: Notebooks directory not found: {notebooks_dir}")
        return
    
    # Check if Jupyter is installed
    try:
        subprocess.run(["jupyter", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Jupyter is not installed. Please install it with:")
        print("pip install jupyter")
        return
    
    # Start the Jupyter notebook server
    print(f"Starting Jupyter notebook server in {notebooks_dir}...")
    os.chdir(notebooks_dir)
    subprocess.run(["jupyter", "notebook"])


if __name__ == "__main__":
    main()
