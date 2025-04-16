#!/usr/bin/env python3
"""
Example 06 - Fast treatment of a sequence of PIV images with on-drive storage

This example demonstrates the standard way of treatment of a sequence of PIV images.
The result of treatment is saved in a folder after processing each image pair.
Before treatment of an image pair, the presence of result file is checked and the
PIV analysis is carried out only if this file does not exist. If the file with
results exists, the processing is skipped and results are read from the file.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import pickle
import time

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_pair, piv_params
from pivsuite.visualization import quiver_plot, vector_plot, streamline_plot
from pivsuite.utils.io import load_image


def main():
    """Run the fast sequence with on-drive storage example."""
    print("\nRUNNING EXAMPLE_06_SEQUENCE_FAST_AND_ON_DRIVE...")
    
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
    
    # Create image pairs (1-2, 2-3, 3-4, etc.)
    im1_list = []
    im2_list = []
    
    for i in range(len(image_files) - 1):
        im1_list.append(image_files[i])
        im2_list.append(image_files[i + 1])
    
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
    
    # Create results directory for storing PIV results
    results_dir = output_dir / "piv_results"
    results_dir.mkdir(exist_ok=True)
    
    # Process each image pair
    all_results = []
    
    for i, (im1_path, im2_path) in enumerate(zip(im1_list, im2_list)):
        # Create result filename
        result_file = results_dir / f"result_{i+1:03d}.pkl"
        
        # Check if result file already exists
        if result_file.exists() and not piv_par.get('force_processing', False):
            print(f"Result file {result_file} already exists. Loading results...")
            with open(result_file, 'rb') as f:
                piv_data = pickle.load(f)
        else:
            print(f"Processing image pair {i+1}/{len(im1_list)}: {os.path.basename(im1_path)} - {os.path.basename(im2_path)}")
            
            # Use previous result as initial guess if available
            prev_data = all_results[-1] if all_results else None
            
            # Analyze image pair
            start_time = time.time()
            piv_data, _ = analyze_image_pair(im1_path, im2_path, prev_data, piv_par)
            elapsed_time = time.time() - start_time
            
            print(f"  Processed in {elapsed_time:.2f} seconds")
            print(f"  Grid points: {piv_data['n']}")
            print(f"  Masked vectors: {piv_data['masked_n']}")
            print(f"  Spurious vectors: {piv_data['spurious_n']}")
            
            # Save result to file
            with open(result_file, 'wb') as f:
                pickle.dump(piv_data, f)
        
        # Store result in memory
        all_results.append(piv_data)
        
        # Create quiver plot for this pair
        quiver_plot(
            piv_data,
            scale=1.0,
            color='k',
            background='magnitude',
            title=f'Velocity Field - Pair {i+1}',
            output_path=str(output_dir / f"example06_velocity_{i+1:03d}.png"),
            show=False
        )
    
    # Calculate mean velocity field
    print("\nCalculating mean velocity field...")
    u_sum = np.zeros_like(all_results[0]['u'])
    v_sum = np.zeros_like(all_results[0]['v'])
    
    for result in all_results:
        u_sum += result['u']
        v_sum += result['v']
    
    u_mean = u_sum / len(all_results)
    v_mean = v_sum / len(all_results)
    
    # Create a mean velocity result
    mean_result = all_results[0].copy()
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
        output_path=str(output_dir / "example06_mean_velocity.png"),
        show=False
    )
    
    # Calculate and plot RMS velocity
    print("Calculating RMS velocity...")
    u_squared_sum = np.zeros_like(all_results[0]['u'])
    v_squared_sum = np.zeros_like(all_results[0]['v'])
    
    for result in all_results:
        u_squared_sum += (result['u'] - u_mean)**2
        v_squared_sum += (result['v'] - v_mean)**2
    
    u_rms = np.sqrt(u_squared_sum / len(all_results))
    v_rms = np.sqrt(v_squared_sum / len(all_results))
    
    # Create a RMS velocity result
    rms_result = all_results[0].copy()
    rms_result['u'] = u_rms
    rms_result['v'] = v_rms
    
    # Create vector plot of RMS velocity
    print("Creating vector plot of RMS velocity...")
    vector_plot(
        rms_result,
        component='magnitude',
        cmap='hot',
        title='RMS Velocity',
        output_path=str(output_dir / "example06_rms_velocity.png"),
        show=False
    )
    
    # Create a plot showing the evolution of maximum velocity over time
    print("Creating plot of maximum velocity over time...")
    max_velocities = [np.sqrt(r['u']**2 + r['v']**2).max() for r in all_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(max_velocities) + 1), max_velocities, 'b-o')
    plt.grid(True)
    plt.xlabel('Image Pair')
    plt.ylabel('Maximum Velocity (px)')
    plt.title('Evolution of Maximum Velocity')
    plt.savefig(str(output_dir / "example06_max_velocity_evolution.png"))
    plt.close()
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_06_SEQUENCE_FAST_AND_ON_DRIVE... FINISHED\n")


if __name__ == "__main__":
    main()
