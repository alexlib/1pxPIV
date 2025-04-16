#!/usr/bin/env python3
"""
Example 04 - Treatment of test images from PIV Challenge

This example treats images from the test case A4 of 3rd PIV challenge (Stanislas, 2008).
To visualize more easily results, the four parts of the image are treated separately.

Reference:
Stanislas, M., K. Okamoto, C. J. Kahler, J. Westerweel and F. Scarano, (2008): Main results
of the third international PIV Challenge. Experiments in Fluids, vol. 45, pp. 27-71.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_pair, piv_params
from pivsuite.visualization import quiver_plot, vector_plot, streamline_plot
from pivsuite.utils.io import load_image


def main():
    """Run the PIV Challenge A4 example."""
    print("\nRUNNING EXAMPLE_04_PIV_CHALLENGE_A4...")
    
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test PIVChallenge3A4"
    im1_path = str(data_dir / "A4001_a.tif")
    im2_path = str(data_dir / "A4001_b.tif")
    mask_path = str(data_dir / "Mask.png")
    
    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        print(f"Error: Image files not found. Please check the paths.")
        print("Please download images (case A4) from http://www.pivchallenge.org/pub05/A/A4.zip,")
        print("unzip them and place them to folder ../Data/Test PIVChallenge3A4.")
        return
    
    print(f"Image paths:\n  {im1_path}\n  {im2_path}")
    
    # Load the images
    print("Loading images...")
    im1_orig = load_image(im1_path)
    im2_orig = load_image(im2_path)
    
    # Load mask if it exists
    if os.path.exists(mask_path):
        print(f"Loading mask: {mask_path}")
        mask_orig = load_image(mask_path)
    else:
        print("No mask found, proceeding without mask.")
        mask_orig = None
    
    # Get image dimensions
    height, width = im1_orig.shape
    print(f"Image dimensions: {width}x{height} pixels")
    
    # Define the four quadrants to process separately
    quadrants = [
        {"name": "top-left", "roi": [0, width//2, 0, height//2]},
        {"name": "top-right", "roi": [width//2, width, 0, height//2]},
        {"name": "bottom-left", "roi": [0, width//2, height//2, height]},
        {"name": "bottom-right", "roi": [width//2, width, height//2, height]}
    ]
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Process each quadrant
    all_results = []
    
    for q in quadrants:
        print(f"\nProcessing {q['name']} quadrant...")
        
        # Extract region of interest
        x1, x2, y1, y2 = q['roi']
        im1 = im1_orig[y1:y2, x1:x2]
        im2 = im2_orig[y1:y2, x1:x2]
        
        # Extract mask for this quadrant if available
        if mask_orig is not None:
            mask = mask_orig[y1:y2, x1:x2]
            # Save mask temporarily
            mask_file = str(output_dir / f"temp_mask_{q['name']}.png")
            plt.imsave(mask_file, mask, cmap='gray')
        else:
            mask_file = None
        
        # Set PIV parameters
        piv_par = {}
        
        # Set mask if available
        if mask_file:
            piv_par['im_mask1'] = mask_file
            piv_par['im_mask2'] = mask_file
        
        # Get default parameters
        piv_par = piv_params(None, piv_par, 'defaults')
        
        # Customize parameters for PIV Challenge A4
        piv_par['ia_size_x'] = [64, 32, 16]  # Interrogation area size in x
        piv_par['ia_size_y'] = [64, 32, 16]  # Interrogation area size in y
        piv_par['ia_step_x'] = [32, 16, 8]   # Interrogation area step in x
        piv_par['ia_step_y'] = [32, 16, 8]   # Interrogation area step in y
        piv_par['ia_method'] = 'defspline'   # Interrogation method
        piv_par['cc_window'] = 'welch'       # Window function for cross-correlation
        piv_par['vl_thresh'] = 2.0           # Threshold for median test
        piv_par['rp_method'] = 'linear'      # Method for replacing spurious vectors
        piv_par['sm_method'] = 'gaussian'    # Smoothing method
        
        # Save images temporarily
        im1_file = str(output_dir / f"temp_im1_{q['name']}.tif")
        im2_file = str(output_dir / f"temp_im2_{q['name']}.tif")
        plt.imsave(im1_file, im1, cmap='gray')
        plt.imsave(im2_file, im2, cmap='gray')
        
        # Analyze image pair
        print(f"Analyzing {q['name']} quadrant...")
        start_time = time.time()
        piv_data, _ = analyze_image_pair(im1_file, im2_file, None, piv_par)
        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        # Store results with quadrant information
        piv_data['quadrant'] = q
        all_results.append(piv_data)
        
        # Print some statistics
        print(f"Grid points: {piv_data['n']}")
        print(f"Masked vectors: {piv_data['masked_n']}")
        print(f"Spurious vectors: {piv_data['spurious_n']}")
        
        # Create quiver plot with velocity magnitude background
        print(f"Creating quiver plot for {q['name']} quadrant...")
        quiver_plot(
            piv_data,
            scale=1.0,
            color='k',
            background='magnitude',
            title=f'Velocity Field - {q["name"]} quadrant',
            output_path=str(output_dir / f"example04_quiver_{q['name']}.png"),
            show=False
        )
        
        # Clean up temporary files
        if os.path.exists(im1_file):
            os.remove(im1_file)
        if os.path.exists(im2_file):
            os.remove(im2_file)
        if mask_file and os.path.exists(mask_file):
            os.remove(mask_file)
    
    # Create a combined visualization of all quadrants
    print("\nCreating combined visualization...")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, (q, ax) in enumerate(zip(all_results, axs.flat)):
        # Get velocity fields
        x = q['x'] + q['quadrant']['roi'][0]  # Adjust x coordinates
        y = q['y'] + q['quadrant']['roi'][2]  # Adjust y coordinates
        u = q['u']
        v = q['v']
        
        # Calculate velocity magnitude
        magnitude = np.sqrt(u**2 + v**2)
        
        # Plot velocity magnitude as background
        im = ax.imshow(magnitude, extent=[x.min(), x.max(), y.max(), y.min()], 
                       origin='upper', cmap='jet', aspect='equal')
        
        # Plot velocity vectors
        ax.quiver(x[::3, ::3], y[::3, ::3], u[::3, ::3], v[::3, ::3], 
                  color='k', scale=50, width=0.002)
        
        ax.set_title(f"{q['quadrant']['name']} quadrant")
        ax.set_xlabel('X (px)')
        ax.set_ylabel('Y (px)')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Velocity Magnitude (px)')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(str(output_dir / "example04_combined_quadrants.png"))
    plt.close()
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_04_PIV_CHALLENGE_A4... FINISHED\n")


if __name__ == "__main__":
    main()
