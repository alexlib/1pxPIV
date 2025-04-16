#!/usr/bin/env python3
"""
Script to crop BOS images to keep only the region from pixel 830 to the bottom.
"""

import os
import numpy as np
from skimage import io
from pathlib import Path

def main():
    """Crop BOS images and save the cropped versions."""
    # Define paths
    data_dir = Path("../Data/Test BOS")
    output_dir = Path("../Data/Test BOS Cropped")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Get list of TIF files
    tif_files = list(data_dir.glob("*.tif"))
    
    if not tif_files:
        print(f"No TIF files found in {data_dir}")
        return
    
    print(f"Found {len(tif_files)} TIF files")
    
    # Process each file
    for file_path in tif_files:
        print(f"Processing {file_path.name}...")
        
        # Load image
        img = io.imread(file_path)
        
        # Crop image (keep only rows from 830 to the end)
        cropped_img = img[830:, :]
        
        # Save cropped image
        output_path = output_dir / file_path.name
        io.imsave(output_path, cropped_img)
        print(f"Saved cropped image to {output_path}")
        
        # Print image dimensions
        print(f"Original size: {img.shape}")
        print(f"Cropped size: {cropped_img.shape}")
    
    print("Done!")

if __name__ == "__main__":
    main()
