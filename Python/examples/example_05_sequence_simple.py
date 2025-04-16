#!/usr/bin/env python3
"""
Example 05 - Simple treatment of a sequence of PIV images

This example demonstrates the simplest possible treatment of a sequence of PIV images
using PIVSuite Python. These images show a turbulent flow in a channel below a set of
injection nozzles.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_sequence, piv_params
from pivsuite.visualization import quiver_plot, vector_plot, streamline_plot
from pivsuite.utils.io import load_image


def main():
    """Run the simple sequence example."""
    print("\nRUNNING EXAMPLE_05_SEQUENCE_SIMPLE...")
    
    # Define path to image folder
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test Tububu"
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Get list of images in the folder
    image_files = sorted(glob.glob(str(data_dir / "*.bmp")))
    
    if not image_files:
        print(f"Error: No BMP images found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} images in {data_dir}")
    
    # Create image pairs (1-2, 5-6, 9-10, etc.)
    im1_list = []
    im2_list = []
    
    for i in range(0, len(image_files)-1, 4):
        if i+1 < len(image_files):
            im1_list.append(image_files[i])
            im2_list.append(image_files[i+1])
    
    print(f"Created {len(im1_list)} image pairs for processing")
    
    # Set PIV parameters
    piv_par = {}
    
    # Get default parameters
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Customize parameters for sequence analysis
    piv_par['ia_size_x'] = [32, 16]  # Interrogation area size in x
    piv_par['ia_size_y'] = [32, 16]  # Interrogation area size in y
    piv_par['ia_step_x'] = [16, 8]   # Interrogation area step in x
    piv_par['ia_step_y'] = [16, 8]   # Interrogation area step in y
    piv_par['ia_method'] = 'defspline'  # Interrogation method
    piv_par['cc_window'] = 'welch'      # Window function for cross-correlation
    piv_par['vl_thresh'] = 2.0          # Threshold for median test
    piv_par['rp_method'] = 'linear'     # Method for replacing spurious vectors
    piv_par['sm_method'] = 'gaussian'   # Smoothing method
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Analyze image sequence
    print("Analyzing image sequence...")
    piv_data_seq = analyze_image_sequence(im1_list, im2_list, None, piv_par)
    
    # Process results
    results = piv_data_seq['results']
    print(f"Processed {len(results)} image pairs")
    
    # Calculate mean velocity field
    u_sum = np.zeros_like(results[0]['u'])
    v_sum = np.zeros_like(results[0]['v'])
    
    for result in results:
        u_sum += result['u']
        v_sum += result['v']
    
    u_mean = u_sum / len(results)
    v_mean = v_sum / len(results)
    
    # Create a mean velocity result
    mean_result = results[0].copy()
    mean_result['u'] = u_mean
    mean_result['v'] = v_mean
    
    # Create quiver plot of mean velocity field
    print("Creating quiver plot of mean velocity field...")
    quiver_plot(
        mean_result,
        scale=1.0,
        color='k',
        background='magnitude',
        title='Mean Velocity Field',
        output_path=str(output_dir / "example05_mean_velocity.png"),
        show=False
    )
    
    # Create vector plot of mean velocity magnitude
    print("Creating vector plot of mean velocity magnitude...")
    vector_plot(
        mean_result,
        component='magnitude',
        cmap='jet',
        title='Mean Velocity Magnitude',
        output_path=str(output_dir / "example05_mean_magnitude.png"),
        show=False
    )
    
    # Create streamline plot of mean velocity field
    print("Creating streamline plot of mean velocity field...")
    streamline_plot(
        mean_result,
        density=1.0,
        color='b',
        title='Mean Velocity Streamlines',
        output_path=str(output_dir / "example05_mean_streamlines.png"),
        show=False
    )
    
    # Create animation of velocity fields
    print("Creating animation frames of velocity fields...")
    for i, result in enumerate(results):
        quiver_plot(
            result,
            scale=1.0,
            color='k',
            background='magnitude',
            title=f'Velocity Field - Frame {i+1}',
            output_path=str(output_dir / f"example05_frame_{i+1:03d}.png"),
            show=False
        )
    
    # Calculate and plot RMS velocity
    print("Calculating RMS velocity...")
    u_squared_sum = np.zeros_like(results[0]['u'])
    v_squared_sum = np.zeros_like(results[0]['v'])
    
    for result in results:
        u_squared_sum += (result['u'] - u_mean)**2
        v_squared_sum += (result['v'] - v_mean)**2
    
    u_rms = np.sqrt(u_squared_sum / len(results))
    v_rms = np.sqrt(v_squared_sum / len(results))
    
    # Create a RMS velocity result
    rms_result = results[0].copy()
    rms_result['u'] = u_rms
    rms_result['v'] = v_rms
    
    # Create vector plot of RMS velocity
    print("Creating vector plot of RMS velocity...")
    vector_plot(
        rms_result,
        component='magnitude',
        cmap='hot',
        title='RMS Velocity',
        output_path=str(output_dir / "example05_rms_velocity.png"),
        show=False
    )
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_05_SEQUENCE_SIMPLE... FINISHED\n")


if __name__ == "__main__":
    main()
