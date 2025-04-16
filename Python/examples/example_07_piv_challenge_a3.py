#!/usr/bin/env python3
"""
Example 07 - Treating test case A3 from 3rd PIV Challenge

This example treats the test case A3 from 3rd PIV challenge (Stanislas, 2008).
Mean and rms velocities are computed, velocity PDF is determined and wavenumber
spectra is calculated.

Another shown feature is the possibility to add iterations during the processing.
Initially, the data are processed with three passes (with IA size 64x64, 32x32 and
16x16 pixels). Additional two passes (with 8x8 pixels interrogation size and 4x4 px
spacing) are then performed. Effect of number of passes on velocity PDF and spectra
is shown.

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
    """Run the PIV Challenge A3 example."""
    print("\nRUNNING EXAMPLE_07_PIV_CHALLENGE_A3...")
    
    # Define path to image folder
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test PIVChallenge3A3"
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        print("Please download images (case A3) from http://www.pivchallenge.org/pub05/A/A3.zip,")
        print("unzip them and place them to folder ../Data/Test PIVChallenge3A3.")
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
    
    # Create results directories for storing PIV results
    results_dir1 = output_dir / "pivOut1"
    results_dir1.mkdir(exist_ok=True)
    results_dir2 = output_dir / "pivOut2"
    results_dir2.mkdir(exist_ok=True)
    
    # Set PIV parameters for first 3 passes
    piv_par1 = {}
    
    # Customize parameters for first 3 passes
    piv_par1['ia_size_x'] = [64, 32, 16]  # Interrogation area size in x
    piv_par1['ia_size_y'] = [64, 32, 16]  # Interrogation area size in y
    piv_par1['ia_step_x'] = [32, 16, 8]   # Interrogation area step in x
    piv_par1['ia_step_y'] = [32, 16, 8]   # Interrogation area step in y
    piv_par1['ia_method'] = 'defspline'   # Interrogation method
    piv_par1['cc_window'] = 'welch'       # Window function for cross-correlation
    piv_par1['vl_thresh'] = 2.0           # Threshold for median test
    piv_par1['rp_method'] = 'linear'      # Method for replacing spurious vectors
    piv_par1['sm_method'] = 'gaussian'    # Smoothing method
    
    # Get default parameters
    piv_par1 = piv_params(None, piv_par1, 'defaults')
    
    # Analyze image sequence with first 3 passes
    print("\nRunning first 3 passes of PIV processing...")
    piv_data1_results = []
    
    for i, (im1_path, im2_path) in enumerate(zip(a_images, b_images)):
        # Create result filename
        result_file = results_dir1 / f"result_{i+1:03d}.npy"
        
        # Check if result file already exists
        if result_file.exists() and not piv_par1.get('force_processing', False):
            print(f"Result file {result_file} already exists. Loading results...")
            piv_data = np.load(result_file, allow_pickle=True).item()
        else:
            print(f"Processing image pair {i+1}/{len(a_images)}: {os.path.basename(im1_path)} - {os.path.basename(im2_path)}")
            
            # Analyze image pair
            start_time = time.time()
            piv_data, _ = analyze_image_pair(im1_path, im2_path, None, piv_par1)
            elapsed_time = time.time() - start_time
            
            print(f"  Processed in {elapsed_time:.2f} seconds")
            print(f"  Grid points: {piv_data['n']}")
            print(f"  Masked vectors: {piv_data['masked_n']}")
            print(f"  Spurious vectors: {piv_data['spurious_n']}")
            
            # Save result to file
            np.save(result_file, piv_data)
        
        # Store result in memory
        piv_data1_results.append(piv_data)
        
        # Create quiver plot for this pair
        quiver_plot(
            piv_data,
            scale=1.0,
            color='k',
            background='magnitude',
            title=f'Velocity Field (16x16 px) - Pair {i+1}',
            output_path=str(output_dir / f"example07_velocity1_{i+1:03d}.png"),
            show=False
        )
    
    # Set PIV parameters for additional 2 passes
    piv_par2 = {}
    
    # Customize parameters for additional 2 passes
    piv_par2['ia_size_x'] = [8, 8]  # Interrogation area size in x
    piv_par2['ia_size_y'] = [8, 8]  # Interrogation area size in y
    piv_par2['ia_step_x'] = [4, 4]  # Interrogation area step in x
    piv_par2['ia_step_y'] = [4, 4]  # Interrogation area step in y
    piv_par2['ia_method'] = 'defspline'  # Interrogation method
    piv_par2['cc_method'] = 'dcn'        # Cross-correlation method (direct convolution)
    piv_par2['cc_window'] = 'welch'      # Window function for cross-correlation
    piv_par2['vl_thresh'] = 2.0          # Threshold for median test
    piv_par2['rp_method'] = 'linear'     # Method for replacing spurious vectors
    piv_par2['sm_method'] = 'gaussian'   # Smoothing method
    
    # Get default parameters
    piv_par2 = piv_params(None, piv_par2, 'defaults')
    
    # Analyze image sequence with additional 2 passes
    print("\nRunning additional 2 passes of PIV processing...")
    piv_data2_results = []
    
    for i, (im1_path, im2_path) in enumerate(zip(a_images, b_images)):
        # Create result filename
        result_file = results_dir2 / f"result_{i+1:03d}.npy"
        
        # Check if result file already exists
        if result_file.exists() and not piv_par2.get('force_processing', False):
            print(f"Result file {result_file} already exists. Loading results...")
            piv_data = np.load(result_file, allow_pickle=True).item()
        else:
            print(f"Processing image pair {i+1}/{len(a_images)}: {os.path.basename(im1_path)} - {os.path.basename(im2_path)}")
            
            # Use previous result as initial guess
            prev_data = piv_data1_results[i]
            
            # Analyze image pair
            start_time = time.time()
            piv_data, _ = analyze_image_pair(im1_path, im2_path, prev_data, piv_par2)
            elapsed_time = time.time() - start_time
            
            print(f"  Processed in {elapsed_time:.2f} seconds")
            print(f"  Grid points: {piv_data['n']}")
            print(f"  Masked vectors: {piv_data['masked_n']}")
            print(f"  Spurious vectors: {piv_data['spurious_n']}")
            
            # Save result to file
            np.save(result_file, piv_data)
        
        # Store result in memory
        piv_data2_results.append(piv_data)
        
        # Create quiver plot for this pair
        quiver_plot(
            piv_data,
            scale=1.0,
            color='k',
            background='magnitude',
            title=f'Velocity Field (8x8 px) - Pair {i+1}',
            output_path=str(output_dir / f"example07_velocity2_{i+1:03d}.png"),
            show=False
        )
    
    # Show results
    print("\nComputing statistics and creating plots...")
    
    # Compute statistics for the velocity fields after 16x16 px
    u1_all = np.array([r['u'] for r in piv_data1_results])
    v1_all = np.array([r['v'] for r in piv_data1_results])
    
    u1_mean = np.mean(u1_all)
    v1_mean = np.mean(v1_all)
    u1_std = np.std(u1_all)
    v1_std = np.std(v1_all)
    
    # Compute statistics for the velocity fields after 8x8 px
    u2_all = np.array([r['u'] for r in piv_data2_results])
    v2_all = np.array([r['v'] for r in piv_data2_results])
    
    u2_mean = np.mean(u2_all)
    v2_mean = np.mean(v2_all)
    u2_std = np.std(u2_all)
    v2_std = np.std(v2_all)
    
    # Print results
    print("\nStatistics (16x16 px): mean(U) = {:.4f}, mean(V) = {:.4f}, std(U) = {:.4f}, std(V) = {:.4f}".format(
        u1_mean, -v1_mean, u1_std, v1_std))
    print("Statistics (8x8 px):   mean(U) = {:.4f}, mean(V) = {:.4f}, std(U) = {:.4f}, std(V) = {:.4f}".format(
        u2_mean, -v2_mean, u2_std, v2_std))
    print("Reference:             mean(U) = {:.4f}, mean(V) = {:.4f}, std(U) = {:.4f}, std(V) = {:.4f}".format(
        -0.0375, -0.0003, 0.6414, 0.5527))
    
    # Compute histogram of u' and show it
    # Define bin range of histogram
    bin_ranges = np.arange(-3, 3.01, 0.02)
    
    # Compute and normalize histogram for 16x16 px
    u1_prime = u1_all - u1_mean
    hist1, _ = np.histogram(u1_prime.flatten(), bins=bin_ranges)
    hist1 = hist1 / (np.sum(hist1) * (bin_ranges[1] - bin_ranges[0]))
    
    # Compute and normalize histogram for 8x8 px
    u2_prime = u2_all - u2_mean
    hist2, _ = np.histogram(u2_prime.flatten(), bins=bin_ranges)
    hist2 = hist2 / (np.sum(hist2) * (bin_ranges[1] - bin_ranges[0]))
    
    # Plot histograms
    plt.figure(figsize=(10, 6))
    plt.plot(bin_ranges[:-1], hist1, '-r', label='16x16 px')
    plt.plot(bin_ranges[:-1], hist2, '-k', label='8x8 px')
    plt.legend()
    plt.xlabel('displacement U (px)')
    plt.ylabel('PDF (a.u.)')
    plt.title('Velocity PDF - compare to Fig. 18 in [6]')
    plt.grid(True)
    plt.savefig(str(output_dir / "example07_velocity_pdf.png"))
    plt.close()
    
    # Compute power spectra of u'
    # Spectrum from the velocity field after 16x16 px
    u1_prime = u1_all - u1_mean
    u1_prime = u1_prime[:, ::1, :]  # Reduce amount of velocity data
    
    u1_spectra = []
    for ky in range(u1_prime.shape[1]):
        for kt in range(u1_prime.shape[0]):
            u1_spectra.append(np.abs(np.fft.fft(u1_prime[kt, ky, :]))**2)
    
    u1_spectra = np.mean(u1_spectra, axis=0)
    u1_spectra = u1_spectra[:len(u1_spectra)//2]
    
    # Determine wavenumber corresponding to the spectra
    x1 = piv_data1_results[0]['x']
    dk1 = 1 / (x1[0, -1] - x1[0, 0])
    k1 = np.arange(len(u1_spectra)) * dk1
    
    # Normalize spectrum
    u1_spectra = u1_spectra / np.sum(u1_spectra) * np.std(u1_prime)**2 / (2 * np.pi * dk1)
    
    # Spectrum from the velocity field after 8x8 px
    u2_prime = u2_all - u2_mean
    u2_prime = u2_prime[:, ::4, :]  # Reduce amount of velocity data
    
    u2_spectra = []
    for ky in range(u2_prime.shape[1]):
        for kt in range(u2_prime.shape[0]):
            u2_spectra.append(np.abs(np.fft.fft(u2_prime[kt, ky, :]))**2)
    
    u2_spectra = np.mean(u2_spectra, axis=0)
    u2_spectra = u2_spectra[:len(u2_spectra)//2]
    
    # Determine wavenumber corresponding to the spectra
    x2 = piv_data2_results[0]['x']
    dk2 = 1 / (x2[0, -1] - x2[0, 0])
    k2 = np.arange(len(u2_spectra)) * dk2
    
    # Normalize spectrum
    u2_spectra = u2_spectra / np.sum(u2_spectra) * np.std(u2_prime)**2 / (2 * np.pi * dk2)
    
    # Plot spectra
    plt.figure(figsize=(10, 6))
    plt.loglog(2 * np.pi * k1, u1_spectra, '-r', label='16x16 px')
    plt.loglog(2 * np.pi * k2, u2_spectra, '-k', label='8x8 px')
    plt.legend()
    plt.xlabel('k_x (1/px)')
    plt.ylabel('E (a.u.)')
    plt.xlim([5e-3, 1])
    plt.ylim([1e-4, 10])
    plt.title('Velocity spectra - compare to Figs. 3c and 16 in [6]')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.savefig(str(output_dir / "example07_velocity_spectra.png"))
    plt.close()
    
    print("All plots saved to the output directory.")
    print("EXAMPLE_07_PIV_CHALLENGE_A3... FINISHED\n")


if __name__ == "__main__":
    main()
