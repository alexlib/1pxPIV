#!/usr/bin/env python3
"""
Example script for analyzing a pair of cropped BOS images.

This script demonstrates how to use the PIVSuite Python package to analyze
cropped BOS images and visualize density gradients in fluids.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.bos import (
    analyze_bos_image_pair,
    plot_bos_results,
    plot_bos_quiver_only,
    compute_bos_magnitude,
    compute_bos_divergence,
    compute_bos_curl
)


def main():
    """Run the BOS cropped image pair example."""
    print("\nRUNNING EXAMPLE_BOS_CROPPED_IMAGE_PAIR...")
    t_start = time.time()
    
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test BOS Cropped"
    im1_path = str(data_dir / "11-49-28.000-4.tif")
    im2_path = str(data_dir / "11-49-28.000-6.tif")
    
    # Set parameters for analysis
    window_size = 32
    step_size = 16
    scale = 0.5  # Process at 50% of original size for faster computation
    
    # Analyze BOS image pair
    results = analyze_bos_image_pair(
        im1_path=im1_path,
        im2_path=im2_path,
        window_size=window_size,
        step_size=step_size,
        scale=scale
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Plot results
    print("Creating quiver plot over the image...")
    plot_bos_results(
        results=results,
        output_path=str(output_dir / "bos_cropped_quiver_plot.png"),
        quiver_scale=15.0,
        arrow_width=2.0,
        arrow_headsize=1.0,
        show_background=True
    )
    
    # Plot quiver only
    print("Creating quiver-only plot...")
    plot_bos_quiver_only(
        results=results,
        output_path=str(output_dir / "bos_cropped_quiver_only.png"),
        quiver_scale=15.0,
        arrow_width=2.0,
        arrow_headsize=1.0
    )
    
    # Compute and plot additional fields
    print("Computing and plotting additional fields...")
    
    # Compute magnitude
    magnitude = compute_bos_magnitude(results)
    
    # Create figure for magnitude
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(results['x'], results['y'], magnitude, cmap='jet', shading='auto')
    plt.colorbar(label='Displacement Magnitude')
    plt.title('BOS Displacement Magnitude (Cropped)', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(str(output_dir / "bos_cropped_magnitude.png"), dpi=300)
    
    # Compute curl (vorticity)
    curl = compute_bos_curl(results)
    
    # Create figure for curl
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(results['x'], results['y'], curl, cmap='RdBu_r', shading='auto')
    plt.colorbar(label='Curl')
    plt.title('BOS Displacement Curl (Cropped)', fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(str(output_dir / "bos_cropped_curl.png"), dpi=300)
    
    print(f"EXAMPLE_BOS_CROPPED_IMAGE_PAIR... FINISHED in {time.time() - t_start:.1f} sec.\n")


if __name__ == "__main__":
    main()
