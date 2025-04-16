"""
Pytest configuration file for PIVSuite Python tests.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import the pivsuite package
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_image_pair():
    """Create a pair of synthetic PIV images with a known displacement."""
    # Parameters
    size = 128
    particle_density = 0.01
    particle_size = 2.0
    displacement = 5.0
    
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
    
    return im1, im2, displacement


@pytest.fixture
def output_dir():
    """Create a temporary output directory for test results."""
    output_dir = Path(__file__).parent.parent / "output" / "test_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir
