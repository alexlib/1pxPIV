"""
Single-pixel PIV validation module for PIVSuite Python

This module implements the validation functions for single-pixel PIV analysis.
It corresponds to the pivSinglepixValidate.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import ndimage


def singlepix_validate(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate displacement field from single-pixel PIV analysis.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with validated displacement field
    """
    # Get parameters
    validation_method = piv_params.get('sp_validation_method', 'median')
    threshold = piv_params.get('sp_validation_threshold', 2.0)
    kernel_size = piv_params.get('sp_validation_kernel_size', 5)
    
    # Get displacement field
    u = piv_data['u']
    v = piv_data['v']
    status = piv_data['status']
    valid_vectors = piv_data['valid_vectors']
    
    # Apply validation
    if validation_method.lower() == 'median':
        valid_vectors = median_validation(u, v, valid_vectors, threshold, kernel_size)
    elif validation_method.lower() == 'none':
        pass  # No validation
    
    # Update status array
    status[~valid_vectors & (status == 0)] = 2  # Mark as invalid by validation
    
    # Store results in piv_data
    piv_data['status'] = status
    piv_data['valid_vectors'] = valid_vectors
    
    # Count invalid vectors
    piv_data['invalid_vectors_n'] = np.sum(~valid_vectors)
    
    return piv_data


def median_validation(
    u: np.ndarray,
    v: np.ndarray,
    valid_vectors: np.ndarray,
    threshold: float,
    kernel_size: int
) -> np.ndarray:
    """
    Apply median validation to displacement field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of displacement field
    v : np.ndarray
        y-component of displacement field
    valid_vectors : np.ndarray
        Mask of valid vectors
    threshold : float
        Threshold for validation
    kernel_size : int
        Size of the kernel for median filter
        
    Returns
    -------
    np.ndarray
        Updated mask of valid vectors
    """
    # Create a copy of the valid vectors mask
    valid_vectors_new = valid_vectors.copy()
    
    # Create a kernel for median filter
    kernel = np.ones((kernel_size, kernel_size))
    kernel[kernel_size//2, kernel_size//2] = 0  # Exclude center point
    
    # Create a copy of the displacement field
    u_valid = u.copy()
    v_valid = v.copy()
    
    # Set invalid vectors to NaN
    u_valid[~valid_vectors] = np.nan
    v_valid[~valid_vectors] = np.nan
    
    # Compute median of displacement field
    u_median = ndimage.median_filter(u_valid, footprint=kernel, mode='reflect')
    v_median = ndimage.median_filter(v_valid, footprint=kernel, mode='reflect')
    
    # Compute residuals
    u_residual = np.abs(u - u_median)
    v_residual = np.abs(v - v_median)
    
    # Compute median of residuals
    u_residual_median = ndimage.median_filter(u_residual, footprint=kernel, mode='reflect')
    v_residual_median = ndimage.median_filter(v_residual, footprint=kernel, mode='reflect')
    
    # Validate vectors
    invalid = (u_residual > threshold * u_residual_median) | (v_residual > threshold * v_residual_median)
    
    # Update valid vectors mask
    valid_vectors_new[invalid & valid_vectors] = False
    
    return valid_vectors_new
