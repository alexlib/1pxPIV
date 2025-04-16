#!/usr/bin/env python3
"""
Script to run the optimized optical flow algorithm on the cropped BOS image pair.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure
from scipy import ndimage

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))


def optimized_lucas_kanade_flow(im1, im2, window_size=12, step_size=6, 
                               gradient_filter_size=2, temporal_filter_size=1,
                               min_eigenvalue=1e-5):
    """
    Optimized Lucas-Kanade optical flow for BOS analysis.
    
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
        Size of the filter for temporal difference calculation
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
    """Run the optimized optical flow algorithm on the cropped BOS image pair."""
    print("\nRunning optimized optical flow on cropped BOS images...")
    
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
    
    # Run optimized optical flow
    print("Running optimized optical flow...")
    results = optimized_lucas_kanade_flow(
        im1_norm, im2_norm,
        window_size=12,
        step_size=6,
        gradient_filter_size=2,
        temporal_filter_size=1,
        min_eigenvalue=1e-5
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
    
    # Create figure for magnitude
    plt.figure(figsize=(15, 10))
    plt.imshow(magnitude_norm, cmap='gray')
    plt.title('Optimized Optical Flow Magnitude', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the magnitude figure
    output_path = str(output_dir / "bos_optimized_flow_magnitude.png")
    plt.savefig(output_path, dpi=600)
    print(f"Magnitude saved to {output_path}")
    
    # Create figure for vector field
    plt.figure(figsize=(15, 10))
    
    # Display the image
    plt.imshow(im1_norm, cmap='gray')
    
    # Downsample the vectors for better visualization
    step = 2
    
    # Plot the quiver
    plt.quiver(x[::step, ::step], y[::step, ::step], 
              u[::step, ::step], v[::step, ::step], 
              color='r', scale=0.1, width=0.001)
    
    plt.title('Optimized Optical Flow Vectors', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the vector field figure
    output_path = str(output_dir / "bos_optimized_flow_vectors.png")
    plt.savefig(output_path, dpi=600)
    print(f"Vector field saved to {output_path}")
    
    # Create figure for vertical component
    plt.figure(figsize=(15, 10))
    
    # Normalize vertical component for visualization
    v_max = max(abs(np.min(v)), abs(np.max(v)))
    v_min = -v_max
    
    # Create the plot with a diverging colormap
    plt.pcolormesh(x, y, v, cmap='RdBu_r', shading='auto', vmin=v_min, vmax=v_max)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Vertical Displacement (pixels)', fontsize=12)
    
    plt.title('Optimized Optical Flow - Vertical Component', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the vertical component figure
    output_path = str(output_dir / "bos_optimized_flow_vertical.png")
    plt.savefig(output_path, dpi=600)
    print(f"Vertical component saved to {output_path}")
    
    # Create figure for horizontal component
    plt.figure(figsize=(15, 10))
    
    # Normalize horizontal component for visualization
    u_max = max(abs(np.min(u)), abs(np.max(u)))
    u_min = -u_max
    
    # Create the plot with a diverging colormap
    plt.pcolormesh(x, y, u, cmap='RdBu_r', shading='auto', vmin=u_min, vmax=u_max)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Horizontal Displacement (pixels)', fontsize=12)
    
    plt.title('Optimized Optical Flow - Horizontal Component', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save the horizontal component figure
    output_path = str(output_dir / "bos_optimized_flow_horizontal.png")
    plt.savefig(output_path, dpi=600)
    print(f"Horizontal component saved to {output_path}")
    
    # Create a combined visualization
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(im1_norm, cmap='gray')
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    # Magnitude
    plt.subplot(2, 2, 2)
    plt.imshow(magnitude_norm, cmap='gray')
    plt.title('Flow Magnitude', fontsize=14)
    plt.axis('off')
    
    # Vertical component
    plt.subplot(2, 2, 3)
    plt.pcolormesh(x, y, v, cmap='RdBu_r', shading='auto', vmin=v_min, vmax=v_max)
    plt.colorbar(label='Vertical Displacement')
    plt.title('Vertical Component', fontsize=14)
    plt.axis('off')
    
    # Horizontal component
    plt.subplot(2, 2, 4)
    plt.pcolormesh(x, y, u, cmap='RdBu_r', shading='auto', vmin=u_min, vmax=u_max)
    plt.colorbar(label='Horizontal Displacement')
    plt.title('Horizontal Component', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the combined figure
    output_path = str(output_dir / "bos_optimized_flow_combined.png")
    plt.savefig(output_path, dpi=600)
    print(f"Combined visualization saved to {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
