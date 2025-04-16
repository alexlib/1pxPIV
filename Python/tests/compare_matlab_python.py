#!/usr/bin/env python3
"""
Comparison script for MATLAB and Python PIV implementations.

This script compares the results of the MATLAB and Python PIV implementations
using the same input images and parameters.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat, savemat

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.core import analyze_image_pair, piv_params


def save_matlab_input(im1, im2, piv_par, output_path):
    """
    Save input data for MATLAB comparison.

    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    piv_par : dict
        PIV parameters
    output_path : str
        Path to save the MATLAB input file
    """
    # Convert parameters to MATLAB format
    matlab_piv_par = {}

    # Map Python parameter names to MATLAB parameter names
    param_map = {
        'ia_size_x': 'iaSizeX',
        'ia_size_y': 'iaSizeY',
        'ia_step_x': 'iaStepX',
        'ia_step_y': 'iaStepY',
        'ia_method': 'iaMethod',
        'cc_window': 'ccWindow',
        'cc_remove_ia_mean': 'ccRemoveIAMean',
        'cc_max_displacement': 'ccMaxDisplacement',
        'vl_thresh': 'vlTresh',
        'vl_eps': 'vlEps',
        'vl_dist': 'vlDist',
        'vl_passes': 'vlPasses',
        'rp_method': 'rpMethod',
        'sm_method': 'smMethod',
        'sm_sigma': 'smSigma',
        'an_n_passes': 'anNpasses'
    }

    # Convert parameters
    for py_name, mat_name in param_map.items():
        if py_name in piv_par:
            matlab_piv_par[mat_name] = piv_par[py_name]

    # Save data to MATLAB file
    savemat(output_path, {
        'im1': im1,
        'im2': im2,
        'pivPar': matlab_piv_par
    })


def load_matlab_results(input_path):
    """
    Load results from MATLAB PIV analysis.

    Parameters
    ----------
    input_path : str
        Path to the MATLAB results file

    Returns
    -------
    dict
        MATLAB PIV results
    """
    # Load MATLAB results
    matlab_data = loadmat(input_path)

    # Extract results
    matlab_results = {}

    # Map MATLAB field names to Python field names
    field_map = {
        'X': 'x',
        'Y': 'y',
        'U': 'u',
        'V': 'v',
        'Status': 'status'
    }

    # Extract fields
    for mat_name, py_name in field_map.items():
        if mat_name in matlab_data['pivData'][0, 0].dtype.names:
            matlab_results[py_name] = matlab_data['pivData'][0, 0][mat_name]

    return matlab_results


def compare_results(python_results, matlab_results, output_path):
    """
    Compare Python and MATLAB PIV results.

    Parameters
    ----------
    python_results : dict
        Python PIV results
    matlab_results : dict
        MATLAB PIV results
    output_path : str
        Path to save the comparison plot
    """
    # Get displacement fields
    u_py = python_results['u']
    v_py = python_results['v']
    u_mat = matlab_results['u']
    v_mat = matlab_results['v']

    # Compute errors
    u_diff = u_py - u_mat
    v_diff = v_py - v_mat

    # Compute statistics
    u_mean_error = np.nanmean(np.abs(u_diff))
    v_mean_error = np.nanmean(np.abs(v_diff))
    u_max_error = np.nanmax(np.abs(u_diff))
    v_max_error = np.nanmax(np.abs(v_diff))

    print(f"Mean absolute error in u: {u_mean_error:.6f}")
    print(f"Mean absolute error in v: {v_mean_error:.6f}")
    print(f"Maximum absolute error in u: {u_max_error:.6f}")
    print(f"Maximum absolute error in v: {v_max_error:.6f}")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Plot Python results
    plt.subplot(2, 3, 1)
    plt.quiver(python_results['x'][::2, ::2], python_results['y'][::2, ::2],
               u_py[::2, ::2], v_py[::2, ::2], scale=50)
    plt.title('Python: Displacement Field')
    plt.axis('equal')

    # Plot MATLAB results
    plt.subplot(2, 3, 2)
    plt.quiver(matlab_results['x'][::2, ::2], matlab_results['y'][::2, ::2],
               u_mat[::2, ::2], v_mat[::2, ::2], scale=50)
    plt.title('MATLAB: Displacement Field')
    plt.axis('equal')

    # Plot difference
    plt.subplot(2, 3, 3)
    plt.quiver(python_results['x'][::2, ::2], python_results['y'][::2, ::2],
               u_diff[::2, ::2], v_diff[::2, ::2], scale=10)
    plt.title('Difference: Displacement Field')
    plt.axis('equal')

    # Plot histograms
    plt.subplot(2, 3, 4)
    plt.hist(u_py.flatten(), bins=20, alpha=0.5, label='Python')
    plt.hist(u_mat.flatten(), bins=20, alpha=0.5, label='MATLAB')
    plt.title('u-component Histogram')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.hist(v_py.flatten(), bins=20, alpha=0.5, label='Python')
    plt.hist(v_mat.flatten(), bins=20, alpha=0.5, label='MATLAB')
    plt.title('v-component Histogram')
    plt.legend()

    # Plot error histogram
    plt.subplot(2, 3, 6)
    plt.hist(u_diff.flatten(), bins=20, alpha=0.5, label='u-error')
    plt.hist(v_diff.flatten(), bins=20, alpha=0.5, label='v-error')
    plt.title('Error Histogram')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Comparison plot saved to {output_path}")


def run_comparison():
    """Run the comparison between MATLAB and Python PIV implementations."""
    print("\nComparing MATLAB and Python PIV implementations...")

    # Define paths
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test PIVChallenge3A1"
    im1_path = str(data_dir / "A1000_a.tif")
    im2_path = str(data_dir / "A1000_b.tif")

    # Check if the image files exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        print(f"Error: Image files not found. Please check the paths.")
        return

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Set PIV parameters
    piv_par = {}
    piv_par = piv_params(None, piv_par, 'defaults')

    # Customize parameters
    piv_par['ia_size_x'] = [32, 16]  # Interrogation area size in x
    piv_par['ia_size_y'] = [32, 16]  # Interrogation area size in y
    piv_par['ia_step_x'] = [16, 8]   # Interrogation area step in x
    piv_par['ia_step_y'] = [16, 8]   # Interrogation area step in y
    piv_par['ia_method'] = 'basic'   # Interrogation method ('basic', 'offset', 'defspline')

    # Load images
    from skimage import io
    im1 = io.imread(im1_path)
    im2 = io.imread(im2_path)

    # Save input data for MATLAB
    matlab_input_path = str(output_dir / "matlab_input.mat")
    save_matlab_input(im1, im2, piv_par, matlab_input_path)

    print(f"MATLAB input data saved to {matlab_input_path}")
    print("Please run the MATLAB script to analyze this data and save the results.")
    print("Then run this script again to compare the results.")

    # Check if MATLAB results exist
    matlab_results_path = str(output_dir / "matlab_results.mat")
    if not os.path.exists(matlab_results_path):
        print(f"MATLAB results file not found: {matlab_results_path}")
        print("Please run the MATLAB script first to generate the results file.")

        # Create a MATLAB script to run the analysis
        matlab_script_path = str(output_dir / "run_matlab_piv.m")
        with open(matlab_script_path, 'w') as f:
            f.write("% MATLAB script to run PIV analysis for comparison\n")
            f.write("% Load input data\n")
            f.write(f"load('{matlab_input_path}');\n\n")
            f.write("% Run PIV analysis\n")
            f.write("[pivData] = pivAnalyzeImagePair(im1, im2, [], pivPar);\n\n")
            f.write("% Save results\n")
            f.write(f"save('{matlab_results_path}', 'pivData');\n")
            f.write("disp('MATLAB PIV analysis completed. Results saved.');\n")

        print(f"MATLAB script created: {matlab_script_path}")
        print("Please run this script in MATLAB to generate the results file.")
        return

    # Run Python PIV analysis
    print("Running Python PIV analysis...")
    python_results, _ = analyze_image_pair(im1, im2, None, piv_par)

    # Load MATLAB results
    print("Loading MATLAB results...")
    matlab_results = load_matlab_results(matlab_results_path)

    # Compare results
    comparison_path = str(output_dir / "matlab_python_comparison.png")
    compare_results(python_results, matlab_results, comparison_path)


if __name__ == "__main__":
    # Run the comparison
    run_comparison()
