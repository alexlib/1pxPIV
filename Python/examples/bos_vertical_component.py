#!/usr/bin/env python3
"""
Script to display the vertical component of BOS displacement field as a colormap.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.bos import analyze_bos_image_pair


def main():
    """Display the vertical component of BOS displacement field."""
    print("\nDisplaying vertical component of BOS displacement field...")
    
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
    
    # Extract data
    x = results['x']
    y = results['y']
    v = results['v']  # Vertical component
    
    # Create figure for vertical component
    plt.figure(figsize=(12, 10))
    
    # Create colormap plot
    # Use a diverging colormap to show positive and negative values
    cmap = plt.cm.RdBu_r
    
    # Compute min and max values for symmetric color scale
    v_max = max(abs(np.min(v)), abs(np.max(v)))
    v_min = -v_max
    
    # Create the plot
    im = plt.pcolormesh(x, y, v, cmap=cmap, shading='auto', vmin=v_min, vmax=v_max)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Vertical Displacement (pixels)', fontsize=12)
    
    # Add title and adjust plot
    plt.title('BOS Vertical Displacement Component', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the figure
    output_path = str(output_dir / "bos_cropped_vertical_component.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # Create a second figure with the vertical component overlaid on the image
    plt.figure(figsize=(12, 10))
    
    # Load the first image
    from skimage import io
    im1 = io.imread(im1_path)
    if scale != 1.0:
        from skimage.transform import resize
        im1 = resize(im1, (int(im1.shape[0] * scale), int(im1.shape[1] * scale)), 
                    anti_aliasing=True, preserve_range=True)
    
    # Display the image
    plt.imshow(im1, cmap='gray')
    
    # Overlay the vertical component with transparency
    im = plt.pcolormesh(x, y, v, cmap=cmap, shading='auto', vmin=v_min, vmax=v_max, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Vertical Displacement (pixels)', fontsize=12)
    
    # Add title and adjust plot
    plt.title('BOS Vertical Displacement Component with Background', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the figure
    output_path = str(output_dir / "bos_cropped_vertical_component_with_background.png")
    plt.savefig(output_path, dpi=300)
    print(f"Plot with background saved to {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
