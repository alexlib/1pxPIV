#!/usr/bin/env python3
"""
Script to tune optical flow parameters for BOS analysis to better match image difference results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure, util
from scipy import ndimage

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))

from pivsuite.bos import analyze_bos_image_pair


def custom_lucas_kanade_flow(im1, im2, window_size=32, step_size=16, 
                            gradient_filter_size=5, temporal_filter_size=0,
                            min_eigenvalue=1e-4):
    """
    Custom implementation of Lucas-Kanade optical flow with more tunable parameters.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    window_size : int
        Size of the window for optical flow calculation
    step_size : int
        Step size between windows
    gradient_filter_size : int
        Size of the filter for gradient calculation
    temporal_filter_size : int
        Size of the filter for temporal difference calculation (0 = no filtering)
    min_eigenvalue : float
        Minimum eigenvalue threshold for the system to be considered well-conditioned
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing optical flow results
    """
    # Get image dimensions
    height, width = im1.shape
    
    # Create grid of points
    x = np.arange(window_size // 2, width - window_size // 2 + 1, step_size)
    y = np.arange(window_size // 2, height - window_size // 2 + 1, step_size)
    X, Y = np.meshgrid(x, y)
    
    # Initialize velocity vectors
    u = np.zeros_like(X, dtype=np.float64)
    v = np.zeros_like(Y, dtype=np.float64)
    
    # Pre-compute gradients for the entire image
    if gradient_filter_size > 0:
        # Use Gaussian derivative for smoother gradients
        Gx = ndimage.gaussian_filter1d(im1, sigma=gradient_filter_size, axis=1, order=1)
        Gy = ndimage.gaussian_filter1d(im1, sigma=gradient_filter_size, axis=0, order=1)
    else:
        # Use simple central differences
        Gx, Gy = np.gradient(im1)
    
    # Compute temporal gradient
    if temporal_filter_size > 0:
        # Smooth images before computing difference
        im1_smooth = ndimage.gaussian_filter(im1, sigma=temporal_filter_size)
        im2_smooth = ndimage.gaussian_filter(im2, sigma=temporal_filter_size)
        Gt = im2_smooth - im1_smooth
    else:
        Gt = im2 - im1
    
    # Calculate optical flow for each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_pos = X[i, j]
            y_pos = Y[i, j]
            
            # Extract windows
            win_gx = Gx[y_pos - window_size // 2:y_pos + window_size // 2,
                       x_pos - window_size // 2:x_pos + window_size // 2]
            win_gy = Gy[y_pos - window_size // 2:y_pos + window_size // 2,
                       x_pos - window_size // 2:x_pos + window_size // 2]
            win_gt = Gt[y_pos - window_size // 2:y_pos + window_size // 2,
                       x_pos - window_size // 2:x_pos + window_size // 2]
            
            # Reshape gradients to vectors
            gx_flat = win_gx.flatten()
            gy_flat = win_gy.flatten()
            gt_flat = win_gt.flatten()
            
            # Create A matrix
            A = np.column_stack((gx_flat, gy_flat))
            
            # Compute ATA matrix
            ATA = A.T @ A
            
            # Check if the system is well-conditioned
            eigenvalues = np.linalg.eigvals(ATA)
            if np.min(eigenvalues) > min_eigenvalue:
                # Calculate flow using least squares
                flow = -np.linalg.inv(ATA) @ A.T @ gt_flat
                u[i, j] = flow[0]
                v[i, j] = flow[1]
    
    # Return results
    return {
        'x': X,
        'y': Y,
        'u': u,
        'v': v
    }


def main():
    """Tune optical flow parameters for BOS analysis."""
    print("\nTuning optical flow parameters for BOS analysis...")
    
    # Define paths to images
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test BOS Cropped"
    im1_path = str(data_dir / "11-49-28.000-4.tif")
    im2_path = str(data_dir / "11-49-28.000-6.tif")
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Load images
    print(f"Loading images from:\n  {im1_path}\n  {im2_path}")
    im1 = io.imread(im1_path)
    im2 = io.imread(im2_path)
    
    # Convert to float for processing
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    
    # Normalize images to [0, 1] range
    im1_norm = im1 / np.max(im1)
    im2_norm = im2 / np.max(im2)
    
    # Compute image difference for reference
    diff = im2_norm - im1_norm
    
    # High-pass filtered difference (reference method)
    diff_highpass = diff - ndimage.gaussian_filter(diff, sigma=5)
    diff_highpass_stretched = exposure.rescale_intensity(
        diff_highpass, 
        in_range=(np.percentile(diff_highpass, 1), np.percentile(diff_highpass, 99))
    )
    
    # Define parameter sets to try
    parameter_sets = [
        # Standard parameters
        {
            'window_size': 32,
            'step_size': 16,
            'gradient_filter_size': 0,
            'temporal_filter_size': 0,
            'min_eigenvalue': 1e-4,
            'title': 'Standard Parameters'
        },
        # Smaller window size
        {
            'window_size': 16,
            'step_size': 8,
            'gradient_filter_size': 0,
            'temporal_filter_size': 0,
            'min_eigenvalue': 1e-4,
            'title': 'Smaller Window (16x16)'
        },
        # Smoothed gradients
        {
            'window_size': 32,
            'step_size': 16,
            'gradient_filter_size': 3,
            'temporal_filter_size': 0,
            'min_eigenvalue': 1e-4,
            'title': 'Smoothed Gradients'
        },
        # Smoothed temporal difference
        {
            'window_size': 32,
            'step_size': 16,
            'gradient_filter_size': 0,
            'temporal_filter_size': 3,
            'min_eigenvalue': 1e-4,
            'title': 'Smoothed Temporal Difference'
        },
        # Combined smoothing
        {
            'window_size': 32,
            'step_size': 16,
            'gradient_filter_size': 3,
            'temporal_filter_size': 3,
            'min_eigenvalue': 1e-4,
            'title': 'Combined Smoothing'
        },
        # Very small window
        {
            'window_size': 8,
            'step_size': 4,
            'gradient_filter_size': 1,
            'temporal_filter_size': 0,
            'min_eigenvalue': 1e-5,
            'title': 'Very Small Window (8x8)'
        },
        # High sensitivity
        {
            'window_size': 32,
            'step_size': 16,
            'gradient_filter_size': 0,
            'temporal_filter_size': 0,
            'min_eigenvalue': 1e-6,
            'title': 'High Sensitivity'
        },
        # Optimized for stripes
        {
            'window_size': 12,
            'step_size': 6,
            'gradient_filter_size': 2,
            'temporal_filter_size': 1,
            'min_eigenvalue': 1e-5,
            'title': 'Optimized for Stripes'
        }
    ]
    
    # Create figure to compare results
    plt.figure(figsize=(20, 20))
    
    # Show original images and difference
    plt.subplot(4, 3, 1)
    plt.imshow(im1_norm, cmap='gray')
    plt.title('Original Image 1', fontsize=14)
    plt.axis('off')
    
    plt.subplot(4, 3, 2)
    plt.imshow(im2_norm, cmap='gray')
    plt.title('Original Image 2', fontsize=14)
    plt.axis('off')
    
    plt.subplot(4, 3, 3)
    plt.imshow(diff_highpass_stretched, cmap='gray')
    plt.title('High-Pass Filtered Difference (Reference)', fontsize=14)
    plt.axis('off')
    
    # Process each parameter set
    for i, params in enumerate(parameter_sets):
        print(f"Processing parameter set {i+1}: {params['title']}")
        
        # Run custom Lucas-Kanade with these parameters
        results = custom_lucas_kanade_flow(
            im1_norm, im2_norm,
            window_size=params['window_size'],
            step_size=params['step_size'],
            gradient_filter_size=params['gradient_filter_size'],
            temporal_filter_size=params['temporal_filter_size'],
            min_eigenvalue=params['min_eigenvalue']
        )
        
        # Extract results
        x = results['x']
        y = results['y']
        u = results['u']
        v = results['v']
        
        # Compute magnitude
        magnitude = np.sqrt(u**2 + v**2)
        
        # Normalize for visualization
        magnitude_norm = exposure.rescale_intensity(
            magnitude, 
            in_range=(np.percentile(magnitude, 1), np.percentile(magnitude, 99))
        )
        
        # Plot results
        plt.subplot(4, 3, i+4)
        plt.imshow(magnitude_norm, cmap='gray')
        plt.title(f"{params['title']}\nWindow: {params['window_size']}x{params['window_size']}, Step: {params['step_size']}", fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison figure
    output_path = str(output_dir / "bos_optical_flow_parameter_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Parameter comparison saved to {output_path}")
    
    # Now create a high-quality version of the best parameter set
    # Based on visual inspection, we'll choose the "Optimized for Stripes" set
    best_params = parameter_sets[7]  # Optimized for Stripes
    
    print(f"Creating high-quality output with best parameters: {best_params['title']}")
    
    # Run custom Lucas-Kanade with best parameters
    results = custom_lucas_kanade_flow(
        im1_norm, im2_norm,
        window_size=best_params['window_size'],
        step_size=best_params['step_size'],
        gradient_filter_size=best_params['gradient_filter_size'],
        temporal_filter_size=best_params['temporal_filter_size'],
        min_eigenvalue=best_params['min_eigenvalue']
    )
    
    # Extract results
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    
    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)
    
    # Create high-quality figure
    plt.figure(figsize=(15, 10))
    
    # Normalize for visualization
    magnitude_norm = exposure.rescale_intensity(
        magnitude, 
        in_range=(np.percentile(magnitude, 1), np.percentile(magnitude, 99))
    )
    
    plt.imshow(magnitude_norm, cmap='gray')
    plt.title(f"Optimized Optical Flow Magnitude\n{best_params['title']}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the high-quality figure
    output_path = str(output_dir / "bos_optical_flow_optimized.png")
    plt.savefig(output_path, dpi=600)
    print(f"Optimized optical flow result saved to {output_path}")
    
    # Also create a version with the optical flow vectors overlaid on the image
    plt.figure(figsize=(15, 10))
    
    # Display the image
    plt.imshow(im1_norm, cmap='gray')
    
    # Downsample the vectors for better visualization
    step = 2
    
    # Plot the quiver
    plt.quiver(x[::step, ::step], y[::step, ::step], 
              u[::step, ::step], v[::step, ::step], 
              color='r', scale=0.1, width=0.001)
    
    plt.title(f"Optimized Optical Flow Vectors\n{best_params['title']}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the vector overlay figure
    output_path = str(output_dir / "bos_optical_flow_vectors.png")
    plt.savefig(output_path, dpi=600)
    print(f"Optical flow vectors saved to {output_path}")
    
    # Create a figure comparing the best optical flow result with the image difference
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(diff_highpass_stretched, cmap='gray')
    plt.title('High-Pass Filtered Difference', fontsize=16)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_norm, cmap='gray')
    plt.title(f"Optimized Optical Flow Magnitude\n{best_params['title']}", fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the comparison figure
    output_path = str(output_dir / "bos_difference_vs_optical_flow.png")
    plt.savefig(output_path, dpi=600)
    print(f"Comparison saved to {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
