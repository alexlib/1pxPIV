"""
Single-pixel PIV replacement module for PIVSuite Python

This module implements the replacement functions for single-pixel PIV analysis.
It corresponds to the pivSinglepixReplace.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import ndimage


def singlepix_replace(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Replace invalid vectors in displacement field.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with replaced vectors
    """
    # Get parameters
    replacement_method = piv_params.get('sp_replacement_method', 'inpaint')
    
    # Get displacement field
    u = piv_data['u']
    v = piv_data['v']
    status = piv_data['status']
    valid_vectors = piv_data['valid_vectors']
    
    # Create a copy of the displacement field
    u_replaced = u.copy()
    v_replaced = v.copy()
    
    # Apply replacement
    if replacement_method.lower() == 'inpaint':
        u_replaced, v_replaced = inpaint_replacement(u, v, valid_vectors)
    elif replacement_method.lower() == 'mean':
        u_replaced, v_replaced = mean_replacement(u, v, valid_vectors)
    elif replacement_method.lower() == 'none':
        pass  # No replacement
    
    # Update status array
    status[~valid_vectors & (status == 2)] = 3  # Mark as replaced
    
    # Store results in piv_data
    piv_data['u_original'] = u
    piv_data['v_original'] = v
    piv_data['u'] = u_replaced
    piv_data['v'] = v_replaced
    piv_data['status'] = status
    
    return piv_data


def inpaint_replacement(
    u: np.ndarray,
    v: np.ndarray,
    valid_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace invalid vectors using inpainting.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of displacement field
    v : np.ndarray
        y-component of displacement field
    valid_vectors : np.ndarray
        Mask of valid vectors
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_replaced, v_replaced: Displacement field with replaced vectors
    """
    # Create a copy of the displacement field
    u_replaced = u.copy()
    v_replaced = v.copy()
    
    # Set invalid vectors to NaN
    u_replaced[~valid_vectors] = np.nan
    v_replaced[~valid_vectors] = np.nan
    
    # Inpaint NaN values
    u_replaced = inpaint_nans(u_replaced)
    v_replaced = inpaint_nans(v_replaced)
    
    return u_replaced, v_replaced


def mean_replacement(
    u: np.ndarray,
    v: np.ndarray,
    valid_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace invalid vectors using mean of valid vectors.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of displacement field
    v : np.ndarray
        y-component of displacement field
    valid_vectors : np.ndarray
        Mask of valid vectors
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_replaced, v_replaced: Displacement field with replaced vectors
    """
    # Create a copy of the displacement field
    u_replaced = u.copy()
    v_replaced = v.copy()
    
    # Compute mean of valid vectors
    u_mean = np.mean(u[valid_vectors])
    v_mean = np.mean(v[valid_vectors])
    
    # Replace invalid vectors with mean
    u_replaced[~valid_vectors] = u_mean
    v_replaced[~valid_vectors] = v_mean
    
    return u_replaced, v_replaced


def inpaint_nans(array: np.ndarray) -> np.ndarray:
    """
    Inpaint NaN values in an array.
    
    Parameters
    ----------
    array : np.ndarray
        Array with NaN values
        
    Returns
    -------
    np.ndarray
        Array with NaN values replaced
    """
    # Create a mask for NaN values
    mask = np.isnan(array)
    
    # If no NaN values, return the array
    if not np.any(mask):
        return array
    
    # Create a copy of the array
    array_inpainted = array.copy()
    
    # Replace NaN values with mean of non-NaN values
    array_inpainted[mask] = np.nanmean(array)
    
    # Smooth the array
    array_inpainted = ndimage.gaussian_filter(array_inpainted, sigma=1.0)
    
    # Restore original values
    array_inpainted[~mask] = array[~mask]
    
    return array_inpainted
