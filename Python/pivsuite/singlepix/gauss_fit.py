"""
Single-pixel PIV Gaussian fitting module for PIVSuite Python

This module implements the Gaussian fitting functions for single-pixel PIV analysis.
It corresponds to the pivSinglepixGaussFit.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from scipy import ndimage, optimize


def singlepix_gauss_fit(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fit Gaussian to cross-correlation peak for sub-pixel displacement.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with sub-pixel displacement
    """
    # Get parameters
    fit_method = piv_params.get('sp_fit_method', 'parabolic')
    
    # Get correlation data
    cc_peak = piv_data['cc_peak']
    cc_x = piv_data['cc_x']
    cc_y = piv_data['cc_y']
    valid_mask = piv_data['valid_mask']
    
    # Initialize arrays for sub-pixel displacement
    u = np.zeros_like(cc_x, dtype=float)
    v = np.zeros_like(cc_y, dtype=float)
    
    # Copy integer displacement
    u[valid_mask] = cc_x[valid_mask]
    v[valid_mask] = cc_y[valid_mask]
    
    # Apply sub-pixel fitting
    if fit_method.lower() == 'parabolic':
        u, v = parabolic_fit(piv_data, piv_params)
    elif fit_method.lower() == 'gaussian':
        u, v = gaussian_fit(piv_data, piv_params)
    
    # Store results in piv_data
    piv_data['u'] = u
    piv_data['v'] = v
    
    return piv_data


def parabolic_fit(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply parabolic fit to cross-correlation peak for sub-pixel displacement.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u, v: Sub-pixel displacement
    """
    # Get correlation data
    cc_peak = piv_data['cc_peak']
    cc_x = piv_data['cc_x']
    cc_y = piv_data['cc_y']
    valid_mask = piv_data['valid_mask']
    
    # Initialize arrays for sub-pixel displacement
    u = np.zeros_like(cc_x, dtype=float)
    v = np.zeros_like(cc_y, dtype=float)
    
    # Copy integer displacement
    u[valid_mask] = cc_x[valid_mask]
    v[valid_mask] = cc_y[valid_mask]
    
    # Get image dimensions
    im_size_y, im_size_x = cc_peak.shape
    
    # Get max displacement
    max_displacement = piv_params.get('sp_max_displacement', 3)
    
    # Create arrays for correlation values
    cc_values = np.zeros((im_size_y, im_size_x, 3, 3))
    
    # Fill correlation values
    for i in range(3):
        for j in range(3):
            dy = i - 1
            dx = j - 1
            
            # Shift the correlation peak
            cc_y_shifted = cc_y + dy
            cc_x_shifted = cc_x + dx
            
            # Check if the shifted peak is within bounds
            valid = (np.abs(cc_y_shifted) <= max_displacement) & (np.abs(cc_x_shifted) <= max_displacement)
            
            # Get correlation values
            for y in range(im_size_y):
                for x in range(im_size_x):
                    if valid[y, x] and valid_mask[y, x]:
                        # Get the correlation value at the shifted peak
                        im2_y = int(y + cc_y_shifted[y, x])
                        im2_x = int(x + cc_x_shifted[y, x])
                        
                        if 0 <= im2_y < im_size_y and 0 <= im2_x < im_size_x:
                            cc_values[y, x, i, j] = cc_peak[im2_y, im2_x]
    
    # Apply parabolic fit
    for y in range(im_size_y):
        for x in range(im_size_x):
            if valid_mask[y, x]:
                # Get the 3x3 correlation values
                cc_3x3 = cc_values[y, x]
                
                # Fit parabola in x direction
                cc_x0 = cc_3x3[1, 0]
                cc_x1 = cc_3x3[1, 1]
                cc_x2 = cc_3x3[1, 2]
                
                # Compute sub-pixel displacement in x
                if cc_x0 < cc_x1 and cc_x2 < cc_x1:
                    dx = 0.5 * (cc_x0 - cc_x2) / (cc_x0 - 2*cc_x1 + cc_x2)
                    u[y, x] += dx
                
                # Fit parabola in y direction
                cc_y0 = cc_3x3[0, 1]
                cc_y1 = cc_3x3[1, 1]
                cc_y2 = cc_3x3[2, 1]
                
                # Compute sub-pixel displacement in y
                if cc_y0 < cc_y1 and cc_y2 < cc_y1:
                    dy = 0.5 * (cc_y0 - cc_y2) / (cc_y0 - 2*cc_y1 + cc_y2)
                    v[y, x] += dy
    
    return u, v


def gaussian_fit(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian fit to cross-correlation peak for sub-pixel displacement.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u, v: Sub-pixel displacement
    """
    # Get correlation data
    cc_peak = piv_data['cc_peak']
    cc_x = piv_data['cc_x']
    cc_y = piv_data['cc_y']
    valid_mask = piv_data['valid_mask']
    
    # Initialize arrays for sub-pixel displacement
    u = np.zeros_like(cc_x, dtype=float)
    v = np.zeros_like(cc_y, dtype=float)
    
    # Copy integer displacement
    u[valid_mask] = cc_x[valid_mask]
    v[valid_mask] = cc_y[valid_mask]
    
    # Get image dimensions
    im_size_y, im_size_x = cc_peak.shape
    
    # Get max displacement
    max_displacement = piv_params.get('sp_max_displacement', 3)
    
    # Create arrays for correlation values
    cc_values = np.zeros((im_size_y, im_size_x, 3, 3))
    
    # Fill correlation values
    for i in range(3):
        for j in range(3):
            dy = i - 1
            dx = j - 1
            
            # Shift the correlation peak
            cc_y_shifted = cc_y + dy
            cc_x_shifted = cc_x + dx
            
            # Check if the shifted peak is within bounds
            valid = (np.abs(cc_y_shifted) <= max_displacement) & (np.abs(cc_x_shifted) <= max_displacement)
            
            # Get correlation values
            for y in range(im_size_y):
                for x in range(im_size_x):
                    if valid[y, x] and valid_mask[y, x]:
                        # Get the correlation value at the shifted peak
                        im2_y = int(y + cc_y_shifted[y, x])
                        im2_x = int(x + cc_x_shifted[y, x])
                        
                        if 0 <= im2_y < im_size_y and 0 <= im2_x < im_size_x:
                            cc_values[y, x, i, j] = cc_peak[im2_y, im2_x]
    
    # Apply Gaussian fit
    for y in range(im_size_y):
        for x in range(im_size_x):
            if valid_mask[y, x]:
                # Get the 3x3 correlation values
                cc_3x3 = cc_values[y, x]
                
                # Fit Gaussian in x direction
                cc_x0 = cc_3x3[1, 0]
                cc_x1 = cc_3x3[1, 1]
                cc_x2 = cc_3x3[1, 2]
                
                # Compute sub-pixel displacement in x
                if cc_x0 < cc_x1 and cc_x2 < cc_x1:
                    try:
                        dx = 0.5 * np.log(cc_x0 / cc_x2) / np.log(cc_x0 * cc_x2 / cc_x1**2)
                        u[y, x] += dx
                    except:
                        pass
                
                # Fit Gaussian in y direction
                cc_y0 = cc_3x3[0, 1]
                cc_y1 = cc_3x3[1, 1]
                cc_y2 = cc_3x3[2, 1]
                
                # Compute sub-pixel displacement in y
                if cc_y0 < cc_y1 and cc_y2 < cc_y1:
                    try:
                        dy = 0.5 * np.log(cc_y0 / cc_y2) / np.log(cc_y0 * cc_y2 / cc_y1**2)
                        v[y, x] += dy
                    except:
                        pass
    
    return u, v
