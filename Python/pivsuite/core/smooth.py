"""
Smoothing module for PIVSuite Python

This module handles the smoothing of velocity fields.
It corresponds to the pivSmooth.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import ndimage


def smooth_velocity(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Smooth the velocity field.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with smoothed vectors
    """
    # Get velocity fields
    u = piv_data['u']
    v = piv_data['v']
    
    # Get smoothing parameters
    sm_method = piv_params.get('sm_method', 'none')
    
    # If no smoothing is needed, return unchanged
    if sm_method.lower() == 'none':
        return piv_data
    
    # Create a copy of the velocity fields
    u_smoothed = np.copy(u)
    v_smoothed = np.copy(v)
    
    # Apply smoothing
    if sm_method.lower() == 'gaussian':
        # Use Gaussian smoothing
        sigma = piv_params.get('sm_sigma', 1.0)
        u_smoothed, v_smoothed = gaussian_smoothing(u, v, sigma)
    
    elif sm_method.lower() == 'median':
        # Use median smoothing
        size = piv_params.get('sm_size', 3)
        u_smoothed, v_smoothed = median_smoothing(u, v, size)
    
    elif sm_method.lower() == 'smoothn':
        # Use smoothn algorithm
        s = piv_params.get('sm_s', 0.0)
        u_smoothed, v_smoothed = smoothn_smoothing(u, v, s)
    
    # Store results in piv_data
    piv_data['u_original'] = u
    piv_data['v_original'] = v
    piv_data['u'] = u_smoothed
    piv_data['v'] = v_smoothed
    
    return piv_data


def gaussian_smoothing(
    u: np.ndarray,
    v: np.ndarray,
    sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian smoothing to velocity field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    sigma : float
        Standard deviation of Gaussian kernel
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_smoothed: Smoothed x-component of velocity field
        v_smoothed: Smoothed y-component of velocity field
    """
    # Create a copy of the velocity fields
    u_smoothed = np.copy(u)
    v_smoothed = np.copy(v)
    
    # Create a mask for NaN values
    nan_mask_u = np.isnan(u)
    nan_mask_v = np.isnan(v)
    
    # Replace NaN values with mean
    u_smoothed[nan_mask_u] = np.nanmean(u)
    v_smoothed[nan_mask_v] = np.nanmean(v)
    
    # Apply Gaussian filter
    u_smoothed = ndimage.gaussian_filter(u_smoothed, sigma=sigma)
    v_smoothed = ndimage.gaussian_filter(v_smoothed, sigma=sigma)
    
    # Restore NaN values
    u_smoothed[nan_mask_u] = np.nan
    v_smoothed[nan_mask_v] = np.nan
    
    return u_smoothed, v_smoothed


def median_smoothing(
    u: np.ndarray,
    v: np.ndarray,
    size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply median smoothing to velocity field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    size : int
        Size of median filter kernel
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_smoothed: Smoothed x-component of velocity field
        v_smoothed: Smoothed y-component of velocity field
    """
    # Create a copy of the velocity fields
    u_smoothed = np.copy(u)
    v_smoothed = np.copy(v)
    
    # Create a mask for NaN values
    nan_mask_u = np.isnan(u)
    nan_mask_v = np.isnan(v)
    
    # Replace NaN values with mean
    u_smoothed[nan_mask_u] = np.nanmean(u)
    v_smoothed[nan_mask_v] = np.nanmean(v)
    
    # Apply median filter
    u_smoothed = ndimage.median_filter(u_smoothed, size=size)
    v_smoothed = ndimage.median_filter(v_smoothed, size=size)
    
    # Restore NaN values
    u_smoothed[nan_mask_u] = np.nan
    v_smoothed[nan_mask_v] = np.nan
    
    return u_smoothed, v_smoothed


def smoothn_smoothing(
    u: np.ndarray,
    v: np.ndarray,
    s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply smoothn algorithm to velocity field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    s : float
        Smoothing parameter
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_smoothed: Smoothed x-component of velocity field
        v_smoothed: Smoothed y-component of velocity field
    """
    try:
        # Try to import smoothn
        from ..utils.smoothn import smoothn
        
        # Create a copy of the velocity fields
        u_smoothed = np.copy(u)
        v_smoothed = np.copy(v)
        
        # Apply smoothn algorithm
        u_smoothed, _ = smoothn(u, s=s)
        v_smoothed, _ = smoothn(v, s=s)
        
        return u_smoothed, v_smoothed
    
    except ImportError:
        # If smoothn is not available, use Gaussian smoothing
        print("Warning: smoothn algorithm not available, using Gaussian smoothing instead")
        return gaussian_smoothing(u, v, 1.0)
