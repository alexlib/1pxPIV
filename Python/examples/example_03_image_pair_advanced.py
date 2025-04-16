#!/usr/bin/env python3
"""
Example 03 - Advanced usage of PIVSuite Python

This example demonstrates the advanced usage of PIVSuite Python for obtaining 
the velocity field from a pair of images with advanced parameters for validation,
smoothing, and window functions.
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
    """Run the advanced image pair example."""
    print("\nRUNNING EXAMPLE_03_IMAGE_PAIR_ADVANCED...")
    
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
    
    # Customize parameters for advanced analysis
    # Interrogation area parameters
    piv_par['ia_size_x'] = [64, 32, 16, 8]  # Interrogation area size in x
    piv_par['ia_size_y'] = [64, 32, 16, 8]  # Interrogation area size in y
    piv_par['ia_step_x'] = [32, 16, 8, 4]   # Interrogation area step in x
    piv_par['ia_step_y'] = [32, 16, 8, 4]   # Interrogation area step in y
    piv_par['ia_method'] = 'defspline'      # Interrogation method ('basic', 'offset', 'defspline')
    piv_par['ia_image_to_deform'] = 'both'  # Deform both images
    
    # Cross-correlation parameters
    piv_par['cc_window'] = 'gaussian'       # Window function for cross-correlation
    piv_par['cc_remove_ia_mean'] = 1.0      # Remove IA mean before cross-correlation
    piv_par['cc_max_displacement'] = 0.7    # Maximum allowed displacement as fraction of IA size
    
    # Validation parameters
    piv_par['vl_thresh'] = 1.5              # Threshold for median test (lower = more strict)
    piv_par['vl_eps'] = 0.05                # Epsilon for median test
    piv_par['vl_dist'] = 2                  # Distance for median test
    piv_par['vl_passes'] = 2                # Number of passes for median test
    
    # Replacement parameters
    piv_par['rp_method'] = 'linear'         # Method for replacing spurious vectors
    
    # Smoothing parameters
    piv_par['sm_method'] = 'gaussian'       # Smoothing method
    piv_par['sm_sigma'] = 1.2               # Smoothing parameter
    piv_par['sm_size'] = 7                  # Size of smoothing filter
    
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
        output_path=str(output_dir / "example03_quiver_plot.png"),
        show=False,
        xlabel='position X (px)',
        ylabel='position Y (px)'
    )
    
    # Create quiver plot showing valid and replaced vectors
    print("Creating quiver plot with valid and replaced vectors...")
    fig = plt.figure(figsize=(12, 8))
    
    # Get velocity fields
    x = piv_data['x']
    y = piv_data['y']
    u = piv_data['u']
    v = piv_data['v']
    
    # Get status array
    status = piv_data.get('status', np.zeros_like(u, dtype=np.uint16))
    
    # Create masks for valid and replaced vectors
    valid = (status & 11) == 0  # 11 = 1 + 2 + 8
    replaced = (status & 8) != 0
    
    # Plot velocity magnitude as background
    magnitude = np.sqrt(u**2 + v**2)
    plt.imshow(magnitude, extent=[x.min(), x.max(), y.max(), y.min()], cmap='jet', origin='upper')
    plt.colorbar(label='Velocity Magnitude (px)')
    
    # Plot valid vectors in black
    plt.quiver(x[valid], y[valid], u[valid], v[valid], color='k', scale=50, width=0.002)
    
    # Plot replaced vectors in white
    plt.quiver(x[replaced], y[replaced], u[replaced], v[replaced], color='w', scale=50, width=0.002)
    
    plt.title('Velocity Field (black: valid, white: replaced)')
    plt.xlabel('position X (px)')
    plt.ylabel('position Y (px)')
    plt.tight_layout()
    plt.savefig(str(output_dir / "example03_quiver_valid_replaced.png"))
    plt.close()
    
    # Create vector plot of vorticity
    print("Creating vector plot of vorticity...")
    vector_plot(
        piv_data,
        component='vorticity',
        cmap='RdBu_r',
        title='Vorticity',
        output_path=str(output_dir / "example03_vorticity.png"),
        show=False
    )
    
    # Create streamline plot with colored lines based on velocity magnitude
    print("Creating colored streamline plot...")
    fig = plt.figure(figsize=(10, 8))
    
    # Plot background image
    if im1 is not None:
        plt.imshow(im1, cmap='gray', origin='upper')
    
    # Plot streamlines colored by velocity magnitude
    strm = plt.streamplot(
        x[0, :], y[:, 0], u, v,
        density=1.5,
        color=magnitude,
        cmap='jet',
        linewidth=1.5,
        arrowsize=1.5
    )
    
    plt.colorbar(strm.lines, label='Velocity Magnitude (px)')
    plt.title('Streamlines Colored by Velocity Magnitude')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(str(output_dir / "example03_colored_streamlines.png"))
    plt.close()
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_03_IMAGE_PAIR_ADVANCED... FINISHED\n")


if __name__ == "__main__":
    main()
