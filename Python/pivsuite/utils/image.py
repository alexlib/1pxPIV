"""
Image processing utility functions for PIVSuite Python

This module contains image processing utility functions for PIV analysis.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import ndimage


def apply_min_max_filter(
    image: np.ndarray,
    size: int = 15,
    level: float = 0.1
) -> np.ndarray:
    """
    Apply min-max filter to an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    size : int
        Size of the filter kernel
    level : float
        Contrast level
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    # Create a copy of the image
    filtered = np.copy(image)
    
    # Apply minimum filter
    min_filtered = ndimage.minimum_filter(filtered, size=size)
    
    # Apply maximum filter to the minimum filtered image
    max_min_filtered = ndimage.maximum_filter(min_filtered, size=size)
    
    # Compute the local contrast
    contrast = filtered - max_min_filtered
    
    # Apply the filter
    filtered = filtered + level * contrast
    
    return filtered


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
        
    Returns
    -------
    np.ndarray
        Equalized image
    """
    # Create a copy of the image
    equalized = np.copy(image)
    
    # Compute the histogram
    hist, bins = np.histogram(equalized.flatten(), bins=256, range=(0, 1))
    
    # Compute the cumulative distribution function
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    
    # Use linear interpolation to map the original values to the equalized values
    equalized = np.interp(equalized.flatten(), bins[:-1], cdf).reshape(equalized.shape)
    
    return equalized


def apply_clahe(
    image: np.ndarray,
    kernel_size: int = 8,
    clip_limit: float = 2.0
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image
    kernel_size : int
        Size of the local region for histogram equalization
    clip_limit : float
        Threshold for contrast limiting
        
    Returns
    -------
    np.ndarray
        CLAHE-enhanced image
    """
    try:
        from skimage import exposure
        
        # Create a copy of the image
        enhanced = np.copy(image)
        
        # Normalize image to [0, 1]
        enhanced = (enhanced - np.min(enhanced)) / (np.max(enhanced) - np.min(enhanced))
        
        # Apply CLAHE
        enhanced = exposure.equalize_adapthist(enhanced, kernel_size=kernel_size, clip_limit=clip_limit)
        
        return enhanced
    
    except ImportError:
        print("Warning: skimage.exposure not available, using histogram equalization instead")
        return apply_histogram_equalization(image)
