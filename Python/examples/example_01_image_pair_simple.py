#!/usr/bin/env python3
"""
Example 01 - Simple usage of PIVSuite Python

This example demonstrates the simplest possible use of PIVSuite Python for obtaining 
the velocity field from a pair of images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_pair, piv_params
from pivsuite.visualization import quiver_plot, vector_plot, streamline_plot
from pivsuite.utils.io import load_image


def main():
    """Run the simple image pair example."""
    print("\nRUNNING EXAMPLE_01_IMAGE_PAIR_SIMPLE...")
    
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test von Karman"
    im1_path = str(data_dir / "PIVlab_Karman_01.bmp")
    im2_path = str(data_dir / "PIVlab_Karman_02.bmp")
    mask_path = str(data_dir / "PIVlab_Karman_mask.png")
    
    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path) or not os.path.exists(mask_path):
        print(f"Error: Image files not found. Please check the paths.")
        return
    
    print(f"Image paths:\n  {im1_path}\n  {im2_path}\n  Mask: {mask_path}")
    
    # Set PIV parameters
    piv_par = {}
    
    # Set mask for both images
    piv_par['im_mask1'] = mask_path
    piv_par['im_mask2'] = mask_path
    
    # Get default parameters
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Analyze image pair
    print("Analyzing image pair...")
    piv_data, _ = analyze_image_pair(im1_path, im2_path, None, piv_par)
    
    # Print some statistics
    print(f"Grid points: {piv_data['n']}")
    print(f"Masked vectors: {piv_data['masked_n']}")
    print(f"Spurious vectors: {piv_data['spurious_n']}")
    print(f"Computational time: {sum(piv_data['comp_time']):.2f} seconds")
    
    # Load the first image for background
    im1 = load_image(im1_path)
    
    # Create quiver plot with velocity magnitude background
    print("Creating quiver plot...")
    quiver_plot(
        piv_data,
        scale=1.0,
        color='k',
        background='magnitude',
        title='Particle displacement (px) in a flow around a cylinder',
        output_path=str(output_dir / "example01_quiver_plot.png"),
        show=False,
        xlabel='position X (px)',
        ylabel='position Y (px)'
    )
    
    # Create a cropped quiver plot showing details around the cylinder
    print("Creating cropped quiver plot...")
    quiver_plot(
        piv_data,
        scale=1.0,
        color='k',
        background='magnitude',
        title='Particle displacement (px) in a flow around a cylinder (detail)',
        output_path=str(output_dir / "example01_quiver_plot_detail.png"),
        show=False,
        crop=[350, 850, 230, 530],
        xlabel='position X (px)',
        ylabel='position Y (px)'
    )
    
    # Extract data for velocity profile
    # Interpolate data for desired position
    y = np.arange(230, 531)
    x = np.ones_like(y) * 550
    
    # Get the grid data
    X = piv_data['x']
    Y = piv_data['y']
    U = piv_data['u']
    
    # Interpolate U values at the specified points
    from scipy.interpolate import griddata
    u_interp = griddata((X.flatten(), Y.flatten()), U.flatten(), (x, y), method='cubic')
    
    # Plot velocity profile
    plt.figure(figsize=(8, 6))
    plt.plot(u_interp, y, '-b')
    plt.title('U profile at X = 550 px')
    plt.xlabel('particle displacement U (px)')
    plt.ylabel('position Y (px)')
    plt.grid(True)
    plt.savefig(str(output_dir / "example01_velocity_profile.png"))
    plt.close()
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_01_IMAGE_PAIR_SIMPLE... FINISHED\n")


if __name__ == "__main__":
    main()
