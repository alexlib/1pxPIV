"""
Single-pixel PIV correlation module for PIVSuite Python

This module implements the correlation functions for single-pixel PIV analysis.
It corresponds to the pivSinglepixCorrelate.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import ndimage


def singlepix_correlate(
    im1: np.ndarray,
    im2: np.ndarray,
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute cross-correlation for each pixel in the image pair.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with correlation data
    """
    # Get parameters
    window_size = piv_params.get('sp_window_size', 7)
    max_displacement = piv_params.get('sp_max_displacement', 3)
    
    # Get image dimensions
    im_size_y, im_size_x = im1.shape
    
    # Initialize arrays for correlation results
    cc_peak = np.zeros((im_size_y, im_size_x))
    cc_x = np.zeros((im_size_y, im_size_x))
    cc_y = np.zeros((im_size_y, im_size_x))
    
    # Create a mask for valid pixels (not too close to the border)
    valid_mask = np.ones((im_size_y, im_size_x), dtype=bool)
    valid_mask[:window_size//2, :] = False
    valid_mask[-window_size//2:, :] = False
    valid_mask[:, :window_size//2] = False
    valid_mask[:, -window_size//2:] = False
    
    # Compute cross-correlation for each pixel
    for dy in range(-max_displacement, max_displacement + 1):
        for dx in range(-max_displacement, max_displacement + 1):
            # Shift the second image
            im2_shifted = np.roll(im2, (dy, dx), axis=(0, 1))
            
            # Compute local correlation
            cc = compute_local_correlation(im1, im2_shifted, window_size)
            
            # Update peak and displacement if correlation is higher
            mask = (cc > cc_peak) & valid_mask
            cc_peak[mask] = cc[mask]
            cc_x[mask] = dx
            cc_y[mask] = dy
    
    # Store results in piv_data
    piv_data['cc_peak'] = cc_peak
    piv_data['cc_x'] = cc_x
    piv_data['cc_y'] = cc_y
    piv_data['valid_mask'] = valid_mask
    
    return piv_data


def compute_local_correlation(
    im1: np.ndarray,
    im2: np.ndarray,
    window_size: int
) -> np.ndarray:
    """
    Compute local correlation between two images.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    window_size : int
        Size of the correlation window
        
    Returns
    -------
    np.ndarray
        Local correlation coefficient
    """
    # Create a kernel for local mean and standard deviation
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    
    # Compute local means
    im1_mean = ndimage.convolve(im1, kernel, mode='reflect')
    im2_mean = ndimage.convolve(im2, kernel, mode='reflect')
    
    # Compute local standard deviations
    im1_std = np.sqrt(ndimage.convolve((im1 - im1_mean)**2, kernel, mode='reflect'))
    im2_std = np.sqrt(ndimage.convolve((im2 - im2_mean)**2, kernel, mode='reflect'))
    
    # Compute local correlation
    numerator = ndimage.convolve((im1 - im1_mean) * (im2 - im2_mean), kernel, mode='reflect')
    denominator = im1_std * im2_std
    
    # Avoid division by zero
    denominator[denominator < 1e-10] = 1e-10
    
    # Compute correlation coefficient
    cc = numerator / denominator
    
    return cc
