"""
Tests for the core PIV functionality.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pivsuite.core import analyze_image_pair, piv_params


def test_piv_params():
    """Test the piv_params function."""
    # Test with default parameters
    params = piv_params(None, None, 'defaults')
    
    # Check that default parameters are set
    assert 'ia_size_x' in params
    assert 'ia_size_y' in params
    assert 'ia_step_x' in params
    assert 'ia_step_y' in params
    
    # Test with custom parameters
    custom_params = {
        'ia_size_x': [64, 32],
        'ia_size_y': [64, 32],
        'ia_step_x': [32, 16],
        'ia_step_y': [32, 16],
    }
    params = piv_params(None, custom_params, 'defaults')
    
    # Check that custom parameters are preserved
    assert params['ia_size_x'] == [64, 32]
    assert params['ia_size_y'] == [64, 32]
    assert params['ia_step_x'] == [32, 16]
    assert params['ia_step_y'] == [32, 16]
    
    # Test single pass extraction
    params = piv_params(None, custom_params, 'singlePass', 0)
    
    # Check that parameters for the first pass are extracted
    assert params['ia_size_x'] == 64
    assert params['ia_size_y'] == 64
    assert params['ia_step_x'] == 32
    assert params['ia_step_y'] == 32


def test_analyze_image_pair(synthetic_image_pair, output_dir):
    """Test the analyze_image_pair function with synthetic images."""
    # Get synthetic images
    im1, im2, true_displacement = synthetic_image_pair
    
    # Set PIV parameters
    piv_par = {}
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Customize parameters
    piv_par['ia_size_x'] = [32, 16]  # Interrogation area size in x
    piv_par['ia_size_y'] = [32, 16]  # Interrogation area size in y
    piv_par['ia_step_x'] = [16, 8]   # Interrogation area step in x
    piv_par['ia_step_y'] = [16, 8]   # Interrogation area step in y
    piv_par['ia_method'] = 'basic'   # Interrogation method
    
    # Analyze image pair
    piv_data, _ = analyze_image_pair(im1, im2, None, piv_par)
    
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
    
    # Compute error
    u_error = np.nanmean(np.abs(piv_data['u'] - true_displacement))
    v_error = np.nanmean(np.abs(piv_data['v']))
    
    # Plot results for visual inspection
    plt.figure(figsize=(12, 8))
    
    # Plot synthetic images
    plt.subplot(2, 2, 1)
    plt.imshow(im1, cmap='gray')
    plt.title('Synthetic Image 1')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(im2, cmap='gray')
    plt.title('Synthetic Image 2')
    plt.axis('off')
    
    # Plot displacement field
    plt.subplot(2, 2, 3)
    plt.quiver(piv_data['x'][::2, ::2], piv_data['y'][::2, ::2], 
               piv_data['u'][::2, ::2], piv_data['v'][::2, ::2], scale=50)
    plt.title('Displacement Field')
    plt.axis('equal')
    
    # Plot error
    plt.subplot(2, 2, 4)
    plt.hist(piv_data['u'].flatten(), bins=20, alpha=0.5, label='u')
    plt.hist(piv_data['v'].flatten(), bins=20, alpha=0.5, label='v')
    plt.axvline(true_displacement, color='r', linestyle='--', label='True u')
    plt.axvline(0.0, color='g', linestyle='--', label='True v')
    plt.title('Displacement Histogram')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(str(output_dir / "test_analyze_image_pair.png"), dpi=300)
    plt.close()
    
    # Check that the error is within acceptable limits
    # This is a loose test since the exact error depends on the synthetic image generation
    assert u_error < 5.0, f"u error too large: {u_error}"
    assert v_error < 5.0, f"v error too large: {v_error}"
    
    # Print the errors for reference
    print(f"Mean absolute error in u: {u_error:.6f}")
    print(f"Mean absolute error in v: {v_error:.6f}")
