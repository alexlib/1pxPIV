#!/usr/bin/env python3
"""
Test script for PIV analysis using PIVSuite Python.

This script tests the Python implementation of PIV analysis against
the MATLAB implementation by comparing the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_pair, piv_params


def test_piv_analysis():
    """Test the PIV analysis implementation."""
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test PIVChallenge3A1"
    im1_path = str(data_dir / "A1000_a.tif")
    im2_path = str(data_dir / "A1000_b.tif")
    
    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        pytest.skip("Test PIV images not found. Skipping test.")
    
    # Set PIV parameters
    piv_par = {}
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Customize parameters
    piv_par['ia_size_x'] = [32, 16]  # Interrogation area size in x
    piv_par['ia_size_y'] = [32, 16]  # Interrogation area size in y
    piv_par['ia_step_x'] = [16, 8]   # Interrogation area step in x
    piv_par['ia_step_y'] = [16, 8]   # Interrogation area step in y
    piv_par['ia_method'] = 'basic'   # Interrogation method ('basic', 'offset', 'defspline')
    
    # Analyze image pair
    piv_data, _ = analyze_image_pair(im1_path, im2_path, None, piv_par)
    
    # Check that the results contain the expected fields
    assert 'x' in piv_data
    assert 'y' in piv_data
    assert 'u' in piv_data
    assert 'v' in piv_data
    
    # Check that the shapes are consistent
    assert piv_data['x'].shape == piv_data['y'].shape
    assert piv_data['u'].shape == piv_data['v'].shape
    assert piv_data['x'].shape == piv_data['u'].shape
    
    # Check that the displacement field is not all zeros
    assert np.any(piv_data['u'] != 0)
    assert np.any(piv_data['v'] != 0)
    
    print("PIV analysis test passed!")


def compare_with_matlab_results():
    """
    Compare the Python PIV analysis results with MATLAB results.
    
    This function assumes that the MATLAB results have been saved to a .mat file.
    """
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test PIVChallenge3A1"
    im1_path = str(data_dir / "A1000_a.tif")
    im2_path = str(data_dir / "A1000_b.tif")
    
    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        print("Test PIV images not found. Skipping comparison.")
        return
    
    # Set PIV parameters
    piv_par = {}
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Customize parameters
    piv_par['ia_size_x'] = [32, 16]  # Interrogation area size in x
    piv_par['ia_size_y'] = [32, 16]  # Interrogation area size in y
    piv_par['ia_step_x'] = [16, 8]   # Interrogation area step in x
    piv_par['ia_step_y'] = [16, 8]   # Interrogation area step in y
    piv_par['ia_method'] = 'basic'   # Interrogation method ('basic', 'offset', 'defspline')
    
    # Analyze image pair with Python
    py_data, _ = analyze_image_pair(im1_path, im2_path, None, piv_par)
    
    # Try to load MATLAB results
    try:
        from scipy.io import loadmat
        matlab_file = Path(__file__).parent.parent.parent / "PIVsuite v.0.8.3" / "piv_example_results.mat"
        
        if not matlab_file.exists():
            print(f"MATLAB results file not found: {matlab_file}")
            print("Run the MATLAB example first to generate the results file.")
            return
        
        matlab_data = loadmat(str(matlab_file))
        
        # Extract MATLAB results
        matlab_pivdata = matlab_data['pivData'][0, 0]
        matlab_x = matlab_pivdata['x'][0, 0]
        matlab_y = matlab_pivdata['y'][0, 0]
        matlab_u = matlab_pivdata['u'][0, 0]
        matlab_v = matlab_pivdata['v'][0, 0]
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        # Plot Python results
        plt.subplot(2, 2, 1)
        plt.quiver(py_data['x'], py_data['y'], py_data['u'], py_data['v'], color='r')
        plt.title('Python: Velocity Field')
        plt.axis('equal')
        
        # Plot MATLAB results
        plt.subplot(2, 2, 2)
        plt.quiver(matlab_x, matlab_y, matlab_u, matlab_v, color='b')
        plt.title('MATLAB: Velocity Field')
        plt.axis('equal')
        
        # Plot Python velocity magnitude
        plt.subplot(2, 2, 3)
        py_mag = np.sqrt(py_data['u']**2 + py_data['v']**2)
        plt.pcolormesh(py_data['x'], py_data['y'], py_mag, cmap='jet', shading='auto')
        plt.colorbar()
        plt.title('Python: Velocity Magnitude')
        plt.axis('equal')
        
        # Plot MATLAB velocity magnitude
        plt.subplot(2, 2, 4)
        matlab_mag = np.sqrt(matlab_u**2 + matlab_v**2)
        plt.pcolormesh(matlab_x, matlab_y, matlab_mag, cmap='jet', shading='auto')
        plt.colorbar()
        plt.title('MATLAB: Velocity Magnitude')
        plt.axis('equal')
        
        plt.tight_layout()
        
        # Save the comparison plot
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(str(output_dir / "python_vs_matlab_comparison.png"), dpi=300)
        
        print("Comparison plot saved to output/python_vs_matlab_comparison.png")
        
        # Compute statistics for comparison
        u_diff = np.abs(py_data['u'] - matlab_u)
        v_diff = np.abs(py_data['v'] - matlab_v)
        
        print(f"Mean absolute difference in u: {np.mean(u_diff):.6f}")
        print(f"Mean absolute difference in v: {np.mean(v_diff):.6f}")
        print(f"Maximum absolute difference in u: {np.max(u_diff):.6f}")
        print(f"Maximum absolute difference in v: {np.max(v_diff):.6f}")
        
    except ImportError:
        print("scipy.io not available. Cannot load MATLAB results.")
    except Exception as e:
        print(f"Error comparing with MATLAB results: {e}")


if __name__ == "__main__":
    # Run the test
    test_piv_analysis()
    
    # Compare with MATLAB results if available
    compare_with_matlab_results()
