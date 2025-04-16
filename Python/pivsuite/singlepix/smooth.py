"""
Single-pixel PIV smoothing module for PIVSuite Python

This module implements the smoothing functions for single-pixel PIV analysis.
It corresponds to the pivSinglepixSmooth.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import ndimage


def singlepix_smooth(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Smooth displacement field from single-pixel PIV analysis.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with smoothed displacement field
    """
    # Get parameters
    smoothing_method = piv_params.get('sp_smoothing_method', 'gaussian')
    sigma = piv_params.get('sp_smoothing_sigma', 1.0)
    
    # Get displacement field
    u = piv_data['u']
    v = piv_data['v']
    status = piv_data['status']
    
    # Create a copy of the displacement field
    u_smoothed = u.copy()
    v_smoothed = v.copy()
    
    # Apply smoothing
    if smoothing_method.lower() == 'gaussian':
        u_smoothed, v_smoothed = gaussian_smoothing(u, v, sigma)
    elif smoothing_method.lower() == 'median':
        u_smoothed, v_smoothed = median_smoothing(u, v, int(sigma * 2 + 1))
    elif smoothing_method.lower() == 'none':
        pass  # No smoothing
    
    # Update status array
    status[status == 0] = 4  # Mark as smoothed
    status[status == 3] = 5  # Mark as replaced and smoothed
    
    # Store results in piv_data
    piv_data['u_unsmoothed'] = u
    piv_data['v_unsmoothed'] = v
    piv_data['u'] = u_smoothed
    piv_data['v'] = v_smoothed
    piv_data['status'] = status
    
    return piv_data


def gaussian_smoothing(
    u: np.ndarray,
    v: np.ndarray,
    sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian smoothing to displacement field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of displacement field
    v : np.ndarray
        y-component of displacement field
    sigma : float
        Standard deviation of Gaussian kernel
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_smoothed, v_smoothed: Smoothed displacement field
    """
    # Apply Gaussian filter
    u_smoothed = ndimage.gaussian_filter(u, sigma=sigma)
    v_smoothed = ndimage.gaussian_filter(v, sigma=sigma)
    
    return u_smoothed, v_smoothed


def median_smoothing(
    u: np.ndarray,
    v: np.ndarray,
    kernel_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply median smoothing to displacement field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of displacement field
    v : np.ndarray
        y-component of displacement field
    kernel_size : int
        Size of the kernel for median filter
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_smoothed, v_smoothed: Smoothed displacement field
    """
    # Apply median filter
    u_smoothed = ndimage.median_filter(u, size=kernel_size)
    v_smoothed = ndimage.median_filter(v, size=kernel_size)
    
    return u_smoothed, v_smoothed
