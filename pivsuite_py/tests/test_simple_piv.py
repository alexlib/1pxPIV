#!/usr/bin/env python3
"""
Simple test script for PIV analysis using PIVSuite Python.

This script creates a pair of synthetic images with a known displacement
and tests the PIV analysis functions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_pair, piv_params


def create_synthetic_images(size=256, particle_density=0.01, particle_size=2.0, displacement=5.0):
    """
    Create a pair of synthetic PIV images with a known displacement.

    Parameters
    ----------
    size : int
        Size of the images (square)
    particle_density : float
        Density of particles (fraction of pixels)
    particle_size : float
        Size of particles (standard deviation of Gaussian)
    displacement : float
        Displacement between images (pixels)

    Returns
    -------
    tuple
        im1, im2: Pair of synthetic images
        u_true, v_true: True displacement field
    """
    # Create a grid
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Create a displacement field
    u_true = np.ones((size, size)) * displacement
    v_true = np.zeros((size, size))

    # Create particles
    n_particles = int(size * size * particle_density)
    particles_x = np.random.randint(0, size, n_particles)
    particles_y = np.random.randint(0, size, n_particles)

    # Create the first image
    im1 = np.zeros((size, size))
    for i in range(n_particles):
        # Add a Gaussian particle
        im1 += np.exp(-((X - particles_x[i])**2 + (Y - particles_y[i])**2) / (2 * particle_size**2))

    # Normalize the image
    im1 = im1 / np.max(im1)

    # Create the second image
    im2 = np.zeros((size, size))
    for i in range(n_particles):
        # Add a Gaussian particle with displacement
        im2 += np.exp(-((X - particles_x[i] - displacement)**2 + (Y - particles_y[i])**2) / (2 * particle_size**2))

    # Normalize the image
    im2 = im2 / np.max(im2)

    return im1, im2, u_true, v_true


def test_piv_analysis():
    """Test the PIV analysis implementation with synthetic images."""
    print("\nTesting PIV analysis with synthetic images...")

    # Create synthetic images
    im1, im2, u_true, v_true = create_synthetic_images(
        size=256,
        particle_density=0.01,
        particle_size=2.0,
        displacement=5.0
    )

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
    piv_data, _ = analyze_image_pair(im1, im2, None, piv_par)

    # Get displacement field
    u = piv_data['u']
    v = piv_data['v']

    # Compute error
    u_error = np.nanmean(np.abs(u - 5.0))
    v_error = np.nanmean(np.abs(v))

    print("Mean absolute error in u: {:.6f}".format(u_error))
    print("Mean absolute error in v: {:.6f}".format(v_error))

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Plot results
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
    plt.quiver(piv_data['x'][::2, ::2], piv_data['y'][::2, ::2], u[::2, ::2], v[::2, ::2], scale=50)
    plt.title('Displacement Field')
    plt.axis('equal')

    # Plot error
    plt.subplot(2, 2, 4)
    plt.hist(u.flatten(), bins=20, alpha=0.5, label='u')
    plt.hist(v.flatten(), bins=20, alpha=0.5, label='v')
    plt.axvline(5.0, color='r', linestyle='--', label='True u')
    plt.axvline(0.0, color='g', linestyle='--', label='True v')
    plt.title('Displacement Histogram')
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(output_dir / "synthetic_piv_test.png"), dpi=300)

    print("Test results saved to output/synthetic_piv_test.png")

    # Check if the error is acceptable
    if u_error < 0.5 and v_error < 0.5:
        print("Test PASSED: Error is within acceptable limits.")
        return True
    else:
        print("Test FAILED: Error is too large.")
        return False


if __name__ == "__main__":
    # Run the test
    test_piv_analysis()
