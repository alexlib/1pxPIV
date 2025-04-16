#!/usr/bin/env python3
"""
Test script for the BOS module.

This script tests the Python implementation of the BOS analysis against
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

from pivsuite.bos import (
    analyze_bos_image_pair,
    compute_bos_magnitude,
    compute_bos_curl
)


def test_bos_analysis():
    """Test the BOS analysis implementation."""
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test BOS"
    im1_path = str(data_dir / "11-49-28.000-4.tif")
    im2_path = str(data_dir / "11-49-28.000-6.tif")
    
    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        pytest.skip("Test BOS images not found. Skipping test.")
    
    # Set parameters for analysis
    window_size = 32
    step_size = 16
    scale = 0.25  # Process at 25% of original size for faster computation
    
    # Analyze BOS image pair
    results = analyze_bos_image_pair(
        im1_path=im1_path,
        im2_path=im2_path,
        window_size=window_size,
        step_size=step_size,
        scale=scale
    )
    
    # Check that the results contain the expected fields
    assert 'x' in results
    assert 'y' in results
    assert 'u' in results
    assert 'v' in results
    
    # Check that the shapes are consistent
    assert results['x'].shape == results['y'].shape
    assert results['u'].shape == results['v'].shape
    assert results['x'].shape == results['u'].shape
    
    # Check that the displacement field is not all zeros
    assert np.any(results['u'] != 0)
    assert np.any(results['v'] != 0)
    
    # Compute magnitude and curl
    magnitude = compute_bos_magnitude(results)
    curl = compute_bos_curl(results)
    
    # Check that the magnitude and curl have the expected shapes
    assert magnitude.shape == results['u'].shape
    assert curl.shape == results['u'].shape
    
    # Check that the magnitude is positive
    assert np.all(magnitude >= 0)
    
    # Check that the curl has both positive and negative values
    assert np.any(curl > 0)
    assert np.any(curl < 0)
    
    print("BOS analysis test passed!")


def compare_with_matlab_results():
    """
    Compare the Python BOS analysis results with MATLAB results.
    
    This function assumes that the MATLAB results have been saved to a .mat file.
    """
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test BOS"
    im1_path = str(data_dir / "11-49-28.000-4.tif")
    im2_path = str(data_dir / "11-49-28.000-6.tif")
    
    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        print("Test BOS images not found. Skipping comparison.")
        return
    
    # Set parameters for analysis
    window_size = 32
    step_size = 16
    scale = 0.25  # Process at 25% of original size for faster computation
    
    # Analyze BOS image pair with Python
    py_results = analyze_bos_image_pair(
        im1_path=im1_path,
        im2_path=im2_path,
        window_size=window_size,
        step_size=step_size,
        scale=scale
    )
    
    # Try to load MATLAB results
    try:
        from scipy.io import loadmat
        matlab_file = Path(__file__).parent.parent.parent / "PIVsuite v.0.8.3" / "bos_example_results.mat"
        
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
        
        # Compute magnitude for both results
        py_magnitude = compute_bos_magnitude(py_results)
        matlab_magnitude = np.sqrt(matlab_u**2 + matlab_v**2)
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        # Plot Python results
        plt.subplot(2, 2, 1)
        plt.quiver(py_results['x'], py_results['y'], py_results['u'], py_results['v'], color='r')
        plt.title('Python: Velocity Field')
        plt.axis('equal')
        
        # Plot MATLAB results
        plt.subplot(2, 2, 2)
        plt.quiver(matlab_x, matlab_y, matlab_u, matlab_v, color='b')
        plt.title('MATLAB: Velocity Field')
        plt.axis('equal')
        
        # Plot Python magnitude
        plt.subplot(2, 2, 3)
        plt.pcolormesh(py_results['x'], py_results['y'], py_magnitude, cmap='jet', shading='auto')
        plt.colorbar()
        plt.title('Python: Displacement Magnitude')
        plt.axis('equal')
        
        # Plot MATLAB magnitude
        plt.subplot(2, 2, 4)
        plt.pcolormesh(matlab_x, matlab_y, matlab_magnitude, cmap='jet', shading='auto')
        plt.colorbar()
        plt.title('MATLAB: Displacement Magnitude')
        plt.axis('equal')
        
        plt.tight_layout()
        
        # Save the comparison plot
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(str(output_dir / "python_vs_matlab_comparison.png"), dpi=300)
        
        print("Comparison plot saved to output/python_vs_matlab_comparison.png")
        
        # Compute statistics for comparison
        u_diff = np.abs(py_results['u'] - matlab_u)
        v_diff = np.abs(py_results['v'] - matlab_v)
        
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
    test_bos_analysis()
    
    # Compare with MATLAB results if available
    compare_with_matlab_results()
