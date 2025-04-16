#!/usr/bin/env python3
"""
Script to compare the results of the Matlab and Python implementations of example_01.

This script loads the results from the Matlab and Python implementations of example_01
and compares them quantitatively.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_python_results():
    """Load the results from the Python implementation."""
    output_dir = Path(__file__).parent / "output"
    results_file = output_dir / "example01_results.npz"
    
    if not results_file.exists():
        print(f"Error: Python results file not found: {results_file}")
        print("Please run the Python example first.")
        return None
    
    return np.load(results_file)


def load_matlab_results():
    """Load the results from the Matlab implementation."""
    matlab_dir = Path(__file__).parent.parent / "Matlab"
    results_file = matlab_dir / "example01_results.mat"
    
    if not results_file.exists():
        print(f"Error: Matlab results file not found: {results_file}")
        print("Please run the Matlab example first.")
        return None
    
    try:
        from scipy.io import loadmat
        return loadmat(results_file)
    except ImportError:
        print("Error: scipy.io.loadmat not available. Please install scipy.")
        return None
    except Exception as e:
        print(f"Error loading Matlab results: {e}")
        return None


def compare_results(py_results, mat_results):
    """Compare the results from the Matlab and Python implementations."""
    if py_results is None or mat_results is None:
        return
    
    # Compare grid dimensions
    py_x = py_results['x']
    py_y = py_results['y']
    py_u = py_results['u']
    py_v = py_results['v']
    
    mat_x = mat_results['x']
    mat_y = mat_results['y']
    mat_u = mat_results['u']
    mat_v = mat_results['v']
    
    print("\nGrid dimensions:")
    print(f"Python: {py_x.shape}")
    print(f"Matlab: {mat_x.shape}")
    
    # Compare grid points
    py_n = py_results['n']
    mat_n = mat_results['n'][0, 0]
    
    print("\nGrid points:")
    print(f"Python: {py_n}")
    print(f"Matlab: {mat_n}")
    print(f"Difference: {py_n - mat_n}")
    
    # Compare masked vectors
    py_masked_n = py_results['masked_n']
    mat_masked_n = mat_results['masked_n'][0, 0]
    
    print("\nMasked vectors:")
    print(f"Python: {py_masked_n}")
    print(f"Matlab: {mat_masked_n}")
    print(f"Difference: {py_masked_n - mat_masked_n}")
    
    # Compare spurious vectors
    py_spurious_n = py_results['spurious_n']
    mat_spurious_n = mat_results['spurious_n'][0, 0]
    
    print("\nSpurious vectors:")
    print(f"Python: {py_spurious_n}")
    print(f"Matlab: {mat_spurious_n}")
    print(f"Difference: {py_spurious_n - mat_spurious_n}")
    
    # Compare computational time
    py_comp_time = np.sum(py_results['comp_time'])
    mat_comp_time = np.sum(mat_results['comp_time'])
    
    print("\nComputational time (seconds):")
    print(f"Python: {py_comp_time:.2f}")
    print(f"Matlab: {mat_comp_time:.2f}")
    print(f"Ratio (Python/Matlab): {py_comp_time / mat_comp_time:.2f}")
    
    # Compare velocity fields
    if py_x.shape == mat_x.shape:
        # Compute differences
        u_diff = py_u - mat_u
        v_diff = py_v - mat_v
        
        # Compute statistics
        u_mean_diff = np.mean(u_diff)
        v_mean_diff = np.mean(v_diff)
        u_std_diff = np.std(u_diff)
        v_std_diff = np.std(v_diff)
        u_max_diff = np.max(np.abs(u_diff))
        v_max_diff = np.max(np.abs(v_diff))
        
        print("\nVelocity field differences:")
        print(f"U mean difference: {u_mean_diff:.6f}")
        print(f"V mean difference: {v_mean_diff:.6f}")
        print(f"U standard deviation of difference: {u_std_diff:.6f}")
        print(f"V standard deviation of difference: {v_std_diff:.6f}")
        print(f"U maximum absolute difference: {u_max_diff:.6f}")
        print(f"V maximum absolute difference: {v_max_diff:.6f}")
        
        # Create plots of differences
        output_dir = Path(__file__).parent / "output"
        
        # Plot U difference
        plt.figure(figsize=(10, 8))
        plt.imshow(u_diff, cmap='RdBu_r', origin='upper')
        plt.colorbar(label='U Difference (Python - Matlab)')
        plt.title('U Difference (Python - Matlab)')
        plt.savefig(str(output_dir / "example01_u_diff.png"))
        plt.close()
        
        # Plot V difference
        plt.figure(figsize=(10, 8))
        plt.imshow(v_diff, cmap='RdBu_r', origin='upper')
        plt.colorbar(label='V Difference (Python - Matlab)')
        plt.title('V Difference (Python - Matlab)')
        plt.savefig(str(output_dir / "example01_v_diff.png"))
        plt.close()
        
        # Plot magnitude difference
        mag_diff = np.sqrt(u_diff**2 + v_diff**2)
        plt.figure(figsize=(10, 8))
        plt.imshow(mag_diff, cmap='viridis', origin='upper')
        plt.colorbar(label='Magnitude Difference')
        plt.title('Magnitude Difference')
        plt.savefig(str(output_dir / "example01_mag_diff.png"))
        plt.close()
        
        print(f"\nDifference plots saved to {output_dir}")
    else:
        print("\nWarning: Grid dimensions do not match, cannot compute velocity field differences.")


def main():
    """Main function."""
    print("Comparing Matlab and Python results for example_01...")
    
    # Load results
    py_results = load_python_results()
    mat_results = load_matlab_results()
    
    # Compare results
    compare_results(py_results, mat_results)


if __name__ == "__main__":
    main()
