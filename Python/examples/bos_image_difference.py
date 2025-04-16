#!/usr/bin/env python3
"""
Script to create a difference image from two BOS images with enhanced contrast.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure, util

def main():
    """Create a difference image with enhanced contrast."""
    print("\nCreating difference image with enhanced contrast...")
    
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
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    
    # Normalize images to [0, 1] range
    im1_norm = im1 / np.max(im1)
    im2_norm = im2 / np.max(im2)
    
    # Compute difference
    diff = im2_norm - im1_norm
    
    # Method 1: Simple absolute difference with contrast stretching
    diff_abs = np.abs(diff)
    diff_stretched = exposure.rescale_intensity(diff_abs, in_range=(0, np.percentile(diff_abs, 99.5)))
    
    # Method 2: Enhance small differences
    diff_enhanced = exposure.equalize_adapthist(diff_abs, clip_limit=0.03)
    
    # Method 3: Binary thresholding to highlight strips
    threshold = np.percentile(diff_abs, 95)  # Use 95th percentile as threshold
    diff_binary = diff_abs > threshold
    
    # Method 4: Difference with offset and scaling to show both positive and negative differences
    diff_offset = diff + 0.5  # Shift to make zero difference = 0.5
    diff_offset_stretched = exposure.rescale_intensity(diff_offset, in_range=(0, 1))
    
    # Create figure to display all methods
    plt.figure(figsize=(20, 15))
    
    # Original images
    plt.subplot(3, 3, 1)
    plt.imshow(im1_norm, cmap='gray')
    plt.title('Original Image 1', fontsize=14)
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(im2_norm, cmap='gray')
    plt.title('Original Image 2', fontsize=14)
    plt.axis('off')
    
    # Method 1: Simple absolute difference with contrast stretching
    plt.subplot(3, 3, 3)
    plt.imshow(diff_stretched, cmap='gray')
    plt.title('Absolute Difference (Contrast Stretched)', fontsize=14)
    plt.axis('off')
    
    # Method 2: Enhance small differences
    plt.subplot(3, 3, 4)
    plt.imshow(diff_enhanced, cmap='gray')
    plt.title('Enhanced Difference (CLAHE)', fontsize=14)
    plt.axis('off')
    
    # Method 3: Binary thresholding
    plt.subplot(3, 3, 5)
    plt.imshow(diff_binary, cmap='gray')
    plt.title('Binary Thresholded Difference', fontsize=14)
    plt.axis('off')
    
    # Method 4: Difference with offset
    plt.subplot(3, 3, 6)
    plt.imshow(diff_offset_stretched, cmap='gray')
    plt.title('Offset Difference', fontsize=14)
    plt.axis('off')
    
    # Method 5: Colormap to show positive and negative differences
    plt.subplot(3, 3, 7)
    plt.imshow(diff, cmap='RdBu_r')
    plt.title('Difference (Red-Blue Colormap)', fontsize=14)
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    
    # Method 6: High-pass filtered difference
    from scipy import ndimage
    diff_highpass = diff - ndimage.gaussian_filter(diff, sigma=5)
    diff_highpass_stretched = exposure.rescale_intensity(
        diff_highpass, 
        in_range=(np.percentile(diff_highpass, 1), np.percentile(diff_highpass, 99))
    )
    
    plt.subplot(3, 3, 8)
    plt.imshow(diff_highpass_stretched, cmap='gray')
    plt.title('High-Pass Filtered Difference', fontsize=14)
    plt.axis('off')
    
    # Method 7: Edge detection on difference
    from skimage import feature
    edges = feature.canny(diff_abs, sigma=2)
    
    plt.subplot(3, 3, 9)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection on Difference', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = str(output_dir / "bos_difference_methods.png")
    plt.savefig(output_path, dpi=300)
    print(f"Comparison of methods saved to {output_path}")
    
    # Create a larger, high-quality version of the best method
    # Based on visual inspection, let's choose the high-pass filtered difference
    plt.figure(figsize=(15, 10))
    plt.imshow(diff_highpass_stretched, cmap='gray')
    plt.title('High-Pass Filtered Difference Image', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the high-quality image
    output_path = str(output_dir / "bos_difference_highpass.png")
    plt.savefig(output_path, dpi=600)
    print(f"High-quality difference image saved to {output_path}")
    
    # Also save a version with the binary thresholding for clear strips
    plt.figure(figsize=(15, 10))
    plt.imshow(diff_binary, cmap='gray')
    plt.title('Binary Thresholded Difference Image', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the binary image
    output_path = str(output_dir / "bos_difference_binary.png")
    plt.savefig(output_path, dpi=600)
    print(f"Binary difference image saved to {output_path}")
    
    print("Done!")


if __name__ == "__main__":
    main()
