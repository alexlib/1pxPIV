#!/usr/bin/env python3
"""
Example 08b - Treating test case A2 from 3rd PIV Challenge

This example treats the test case A2 from 3rd PIV challenge (Stanislas, 2008).
Mean and rms velocities are computed, velocity PDF is determined and wavenumber
spectra is calculated.

Reference:
Stanislas, M., K. Okamoto, C. J. Kahler, J. Westerweel and F. Scarano, (2008): Main
results of the third international PIV Challenge. Experiments in Fluids, vol. 45, pp. 27-71.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import time

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_sequence, analyze_image_pair, piv_params
from pivsuite.visualization import quiver_plot, vector_plot, streamline_plot
from pivsuite.utils.io import load_image


def main():
    """Run the PIV Challenge A2 example."""
    print("\nRUNNING EXAMPLE_08B_PIV_CHALLENGE_A2...")
    
    # Define path to image folder
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test PIVChallenge3A2"
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please download images (case A2) from http://www.pivchallenge.org/pub05/A/A2.zip,")
        print("unzip them and place them to folder ../Data/Test PIVChallenge3A2.")
        return
    
    # Get list of images in the folder
    a_images = sorted(glob.glob(str(data_dir / "*a.tif")))
    b_images = sorted(glob.glob(str(data_dir / "*b.tif")))
    
    if not a_images or not b_images:
        print(f"Error: No image pairs found in {data_dir}")
        return
    
    print(f"Found {len(a_images)} 'a' images and {len(b_images)} 'b' images in {data_dir}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create results directory for storing PIV results
    results_dir = output_dir / "pivOut_A2"
    results_dir.mkdir(exist_ok=True)
    
    # Set PIV parameters
    piv_par = {}
    
    # Customize parameters
    piv_par['ia_size_x'] = [64, 32, 16, 8]  # Interrogation area size in x
    piv_par['ia_size_y'] = [64, 32, 16, 8]  # Interrogation area size in y
    piv_par['ia_step_x'] = [32, 16, 8, 4]   # Interrogation area step in x
    piv_par['ia_step_y'] = [32, 16, 8, 4]   # Interrogation area step in y
    piv_par['ia_method'] = 'defspline'      # Interrogation method
    piv_par['cc_window'] = 'welch'          # Window function for cross-correlation
    piv_par['vl_thresh'] = 2.0              # Threshold for median test
    piv_par['rp_method'] = 'linear'         # Method for replacing spurious vectors
    piv_par['sm_method'] = 'gaussian'       # Smoothing method
    
    # Get default parameters
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Analyze image sequence
    print("\nRunning PIV processing...")
    piv_data_results = []
    
    for i, (im1_path, im2_path) in enumerate(zip(a_images, b_images)):
        # Create result filename
        result_file = results_dir / f"result_{i+1:03d}.npy"
        
        # Check if result file already exists
        if result_file.exists() and not piv_par.get('force_processing', False):
            print(f"Result file {result_file} already exists. Loading results...")
            piv_data = np.load(result_file, allow_pickle=True).item()
        else:
            print(f"Processing image pair {i+1}/{len(a_images)}: {os.path.basename(im1_path)} - {os.path.basename(im2_path)}")
            
            # Use previous result as initial guess if available
            prev_data = piv_data_results[-1] if piv_data_results else None
            
            # Analyze image pair
            start_time = time.time()
            piv_data, _ = analyze_image_pair(im1_path, im2_path, prev_data, piv_par)
            elapsed_time = time.time() - start_time
            
            print(f"  Processed in {elapsed_time:.2f} seconds")
            print(f"  Grid points: {piv_data['n']}")
            print(f"  Masked vectors: {piv_data['masked_n']}")
            print(f"  Spurious vectors: {piv_data['spurious_n']}")
            
            # Save result to file
            np.save(result_file, piv_data)
        
        # Store result in memory
        piv_data_results.append(piv_data)
        
        # Create quiver plot for this pair
        quiver_plot(
            piv_data,
            scale=1.0,
            color='k',
            background='magnitude',
            title=f'Velocity Field - Pair {i+1}',
            output_path=str(output_dir / f"example08b_velocity_{i+1:03d}.png"),
            show=False
        )
    
    # Show results
    print("\nComputing statistics and creating plots...")
    
    # Compute statistics for the velocity fields
    u_all = np.array([r['u'] for r in piv_data_results])
    v_all = np.array([r['v'] for r in piv_data_results])
    
    u_mean = np.mean(u_all)
    v_mean = np.mean(v_all)
    u_std = np.std(u_all)
    v_std = np.std(v_all)
    
    # Print results
    print("\nStatistics: mean(U) = {:.4f}, mean(V) = {:.4f}, std(U) = {:.4f}, std(V) = {:.4f}".format(
        u_mean, -v_mean, u_std, v_std))
    print("Reference: mean(U) = {:.4f}, mean(V) = {:.4f}, std(U) = {:.4f}, std(V) = {:.4f}".format(
        0.0, 0.0, 0.5, 0.5))  # Approximate reference values
    
    # Create mean velocity result
    mean_result = piv_data_results[0].copy()
    mean_result['u'] = np.mean(u_all, axis=0)
    mean_result['v'] = np.mean(v_all, axis=0)
    
    # Create quiver plot of mean velocity field
    print("Creating quiver plot of mean velocity field...")
    quiver_plot(
        mean_result,
        scale=1.0,
        color='k',
        background='magnitude',
        title='Mean Velocity Field',
        output_path=str(output_dir / "example08b_mean_velocity.png"),
        show=False
    )
    
    # Create RMS velocity result
    rms_result = piv_data_results[0].copy()
    rms_result['u'] = np.std(u_all, axis=0)
    rms_result['v'] = np.std(v_all, axis=0)
    
    # Create vector plot of RMS velocity
    print("Creating vector plot of RMS velocity...")
    vector_plot(
        rms_result,
        component='magnitude',
        cmap='hot',
        title='RMS Velocity',
        output_path=str(output_dir / "example08b_rms_velocity.png"),
        show=False
    )
    
    # Compute histogram of u' and show it
    # Define bin range of histogram
    bin_ranges = np.arange(-3, 3.01, 0.02)
    
    # Compute and normalize histogram
    u_prime = u_all - u_mean
    hist, _ = np.histogram(u_prime.flatten(), bins=bin_ranges)
    hist = hist / (np.sum(hist) * (bin_ranges[1] - bin_ranges[0]))
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.plot(bin_ranges[:-1], hist, '-k')
    plt.xlabel('displacement U (px)')
    plt.ylabel('PDF (a.u.)')
    plt.title('Velocity PDF - compare to Fig. 14 in [6]')
    plt.grid(True)
    plt.savefig(str(output_dir / "example08b_velocity_pdf.png"))
    plt.close()
    
    # Compute power spectra of u'
    u_prime = u_all - u_mean
    u_prime = u_prime[:, ::1, :]  # Reduce amount of velocity data
    
    u_spectra = []
    for ky in range(u_prime.shape[1]):
        for kt in range(u_prime.shape[0]):
            u_spectra.append(np.abs(np.fft.fft(u_prime[kt, ky, :]))**2)
    
    u_spectra = np.mean(u_spectra, axis=0)
    u_spectra = u_spectra[:len(u_spectra)//2]
    
    # Determine wavenumber corresponding to the spectra
    x = piv_data_results[0]['x']
    dk = 1 / (x[0, -1] - x[0, 0])
    k = np.arange(len(u_spectra)) * dk
    
    # Normalize spectrum
    u_spectra = u_spectra / np.sum(u_spectra) * np.std(u_prime)**2 / (2 * np.pi * dk)
    
    # Plot spectrum
    plt.figure(figsize=(10, 6))
    plt.loglog(2 * np.pi * k, u_spectra, '-k')
    plt.xlabel('k_x (1/px)')
    plt.ylabel('E (a.u.)')
    plt.xlim([5e-3, 1])
    plt.ylim([1e-4, 10])
    plt.title('Velocity spectrum - compare to Fig. 3b in [6]')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(str(output_dir / "example08b_velocity_spectrum.png"))
    plt.close()
    
    # Create a table with statistics
    print("\nCreating table with statistics...")
    
    # Calculate statistics for each image pair
    stats_table = []
    for i, result in enumerate(piv_data_results):
        u_mean_i = np.mean(result['u'])
        v_mean_i = np.mean(result['v'])
        u_std_i = np.std(result['u'])
        v_std_i = np.std(result['v'])
        max_vel_i = np.sqrt(result['u']**2 + result['v']**2).max()
        stats_table.append([i+1, u_mean_i, v_mean_i, u_std_i, v_std_i, max_vel_i])
    
    # Create a figure with the table
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=[[f"{row[0]}", f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}", f"{row[5]:.4f}"] for row in stats_table],
        colLabels=["Pair", "Mean U", "Mean V", "Std U", "Std V", "Max Vel"],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Statistics for each image pair - compare to Table 11 in [6]')
    plt.savefig(str(output_dir / "example08b_statistics_table.png"))
    plt.close()
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_08B_PIV_CHALLENGE_A2... FINISHED\n")


if __name__ == "__main__":
    main()
