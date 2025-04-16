#!/usr/bin/env python3
"""
Script to analyze the results of the Python implementation of example_01.

This script runs the Python example and then analyzes the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import time


def run_python_example():
    """Run the Python example."""
    print("Running Python example...")
    start_time = time.time()
    subprocess.run(["python", "examples/example_01_image_pair_simple.py"], check=True)
    elapsed_time = time.time() - start_time
    print(f"Python example completed in {elapsed_time:.2f} seconds")


def analyze_results():
    """Analyze the results from the Python implementation."""
    output_dir = Path(__file__).parent / "output"
    results_file = output_dir / "example01_results.npz"
    
    if not results_file.exists():
        print(f"Error: Python results file not found: {results_file}")
        return
    
    # Load results
    results = np.load(results_file)
    
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    status = results['status']
    n = results['n']
    masked_n = results['masked_n']
    spurious_n = results['spurious_n']
    comp_time = results['comp_time']
    
    # Print statistics
    print("\nPython Results:")
    print(f"Grid dimensions: {x.shape}")
    print(f"Grid points: {n}")
    print(f"Masked vectors: {masked_n}")
    print(f"Spurious vectors: {spurious_n}")
    print(f"Computational time: {np.sum(comp_time):.2f} seconds")
    
    # Calculate velocity statistics
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    u_std = np.std(u)
    v_std = np.std(v)
    u_min = np.min(u)
    v_min = np.min(v)
    u_max = np.max(u)
    v_max = np.max(v)
    
    print("\nVelocity Statistics:")
    print(f"U mean: {u_mean:.6f}")
    print(f"V mean: {v_mean:.6f}")
    print(f"U standard deviation: {u_std:.6f}")
    print(f"V standard deviation: {v_std:.6f}")
    print(f"U range: [{u_min:.6f}, {u_max:.6f}]")
    print(f"V range: [{v_min:.6f}, {v_max:.6f}]")
    
    # Calculate magnitude statistics
    magnitude = np.sqrt(u**2 + v**2)
    mag_mean = np.mean(magnitude)
    mag_std = np.std(magnitude)
    mag_min = np.min(magnitude)
    mag_max = np.max(magnitude)
    
    print("\nMagnitude Statistics:")
    print(f"Mean: {mag_mean:.6f}")
    print(f"Standard deviation: {mag_std:.6f}")
    print(f"Range: [{mag_min:.6f}, {mag_max:.6f}]")
    
    # Save statistics to a file
    stats_file = output_dir / "example01_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Python Results:\n")
        f.write(f"Grid dimensions: {x.shape}\n")
        f.write(f"Grid points: {n}\n")
        f.write(f"Masked vectors: {masked_n}\n")
        f.write(f"Spurious vectors: {spurious_n}\n")
        f.write(f"Computational time: {np.sum(comp_time):.2f} seconds\n")
        
        f.write("\nVelocity Statistics:\n")
        f.write(f"U mean: {u_mean:.6f}\n")
        f.write(f"V mean: {v_mean:.6f}\n")
        f.write(f"U standard deviation: {u_std:.6f}\n")
        f.write(f"V standard deviation: {v_std:.6f}\n")
        f.write(f"U range: [{u_min:.6f}, {u_max:.6f}]\n")
        f.write(f"V range: [{v_min:.6f}, {v_max:.6f}]\n")
        
        f.write("\nMagnitude Statistics:\n")
        f.write(f"Mean: {mag_mean:.6f}\n")
        f.write(f"Standard deviation: {mag_std:.6f}\n")
        f.write(f"Range: [{mag_min:.6f}, {mag_max:.6f}]\n")
    
    print(f"\nStatistics saved to {stats_file}")


def main():
    """Main function."""
    # Run the Python example
    run_python_example()
    
    # Analyze the results
    analyze_results()


if __name__ == "__main__":
    main()
