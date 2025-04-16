#!/usr/bin/env python3
"""
Basic test script for PIV analysis using PIVSuite Python.

This script creates a pair of synthetic images with a known displacement
and tests a simplified PIV analysis function.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def create_synthetic_images(size=256, particle_density=0.01, particle_size=2.0, displacement=5.0):
    """
    Create a pair of synthetic PIV images with a known displacement.
    """
    # Create a grid
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

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

    return im1, im2


def basic_piv_analysis(im1, im2, window_size=32, step_size=16):
    """
    Perform basic PIV analysis on an image pair using FFT-based cross-correlation.
    """
    # Get image dimensions
    im_size_y, im_size_x = im1.shape

    # Calculate number of interrogation areas
    ia_n_x = (im_size_x - window_size) // step_size + 1
    ia_n_y = (im_size_y - window_size) // step_size + 1

    # Create grid of interrogation area centers
    x = np.arange(window_size/2, window_size/2 + ia_n_x*step_size, step_size)
    y = np.arange(window_size/2, window_size/2 + ia_n_y*step_size, step_size)
    X, Y = np.meshgrid(x, y)

    # Initialize arrays for displacement
    u = np.zeros((ia_n_y, ia_n_x))
    v = np.zeros((ia_n_y, ia_n_x))

    # Process each interrogation area
    for iy in range(ia_n_y):
        for ix in range(ia_n_x):
            # Get the interrogation area from the first image
            ia1_y = int(iy * step_size)
            ia1_x = int(ix * step_size)
            ia1 = im1[ia1_y:ia1_y+window_size, ia1_x:ia1_x+window_size].copy()

            # Get the interrogation area from the second image
            ia2 = im2[ia1_y:ia1_y+window_size, ia1_x:ia1_x+window_size].copy()

            # Remove mean
            ia1 = ia1 - np.mean(ia1)
            ia2 = ia2 - np.mean(ia2)

            # Compute cross-correlation using FFT
            corr = np.fft.fftshift(np.real(np.fft.ifft2(
                np.conj(np.fft.fft2(ia1)) * np.fft.fft2(ia2)
            )))

            # Find the peak
            peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)

            # Compute displacement (peak position relative to center)
            dx = peak_x - window_size // 2
            dy = peak_y - window_size // 2

            # Store displacement
            u[iy, ix] = dx
            v[iy, ix] = dy

            # Sub-pixel refinement using Gaussian peak fit
            if 0 < peak_x < window_size-1 and 0 < peak_y < window_size-1:
                # Fit Gaussian in x direction
                c1 = np.log(corr[peak_y, peak_x-1])
                c2 = np.log(corr[peak_y, peak_x])
                c3 = np.log(corr[peak_y, peak_x+1])
                if c1 < c2 and c3 < c2:  # Check if it's a peak
                    dx_sub = 0.5 * (c1 - c3) / (c1 - 2*c2 + c3)
                    u[iy, ix] += dx_sub

                # Fit Gaussian in y direction
                c1 = np.log(corr[peak_y-1, peak_x])
                c2 = np.log(corr[peak_y, peak_x])
                c3 = np.log(corr[peak_y+1, peak_x])
                if c1 < c2 and c3 < c2:  # Check if it's a peak
                    dy_sub = 0.5 * (c1 - c3) / (c1 - 2*c2 + c3)
                    v[iy, ix] += dy_sub

    return X, Y, u, v


def test_basic_piv():
    """Test the basic PIV analysis implementation with synthetic images."""
    print("\nTesting basic PIV analysis with synthetic images...")

    # Create synthetic images
    im1, im2 = create_synthetic_images(
        size=128,
        particle_density=0.01,
        particle_size=2.0,
        displacement=5.0
    )

    # Analyze image pair
    X, Y, u, v = basic_piv_analysis(im1, im2, window_size=32, step_size=16)

    # Compute error
    u_error = np.mean(np.abs(u - 5.0))
    v_error = np.mean(np.abs(v))

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
    plt.quiver(X, Y, u, v, scale=50)
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
    plt.savefig(str(output_dir / "basic_piv_test.png"), dpi=300)

    print("Test results saved to output/basic_piv_test.png")

    # Check if the error is acceptable
    if u_error < 0.5 and v_error < 0.5:
        print("Test PASSED: Error is within acceptable limits.")
        return True
    else:
        print("Test FAILED: Error is too large.")
        return False


if __name__ == "__main__":
    # Run the test
    test_basic_piv()
