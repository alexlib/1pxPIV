"""
Cross-correlation module for PIVSuite

This module handles the cross-correlation between interrogation areas.
It corresponds to the pivCrossCorr.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Tuple, Dict, Union, Optional, Any
import numba

from .utils import (
    create_window_function, 
    cross_correlate_fft, 
    cross_correlate_direct,
    find_peak_position
)


def compute_cross_correlation(
    ex_im1: np.ndarray, 
    ex_im2: np.ndarray, 
    piv_data: Dict[str, Any], 
    piv_params: Dict[str, Any]
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray]]:
    """
    Compute cross-correlation between interrogation areas.
    
    Parameters
    ----------
    ex_im1 : np.ndarray
        Expanded first image
    ex_im2 : np.ndarray
        Expanded second image
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray]]
        piv_data: Updated PIV results
        cc_peak_im: Cross-correlation peak image (optional)
    """
    # Get parameters
    ia_size_x = piv_params.get('ia_size_x', 32)
    ia_size_y = piv_params.get('ia_size_y', 32)
    cc_method = piv_params.get('cc_method', 'fft')
    cc_window = piv_params.get('cc_window', 'uniform')
    cc_remove_mean = piv_params.get('cc_remove_mean', True)
    cc_max_displacement = piv_params.get('cc_max_displacement', 0.7)
    cc_subpixel_method = piv_params.get('cc_subpixel_method', 'gaussian')
    
    # Get dimensions
    ia_n_y, ia_n_x = piv_data['status'].shape
    
    # Get status array
    status = piv_data['status'].copy()
    
    # Get initial shift of interrogation areas
    ia_u0 = piv_data.get('ia_u0', np.zeros((ia_n_y, ia_n_x)))
    ia_v0 = piv_data.get('ia_v0', np.zeros((ia_n_y, ia_n_x)))
    
    # Create window function
    window = create_window_function(ia_size_x, ia_size_y, cc_window)
    
    # Calculate maximum allowed displacement in pixels
    cc_max_disp_x = int(cc_max_displacement * ia_size_x)
    cc_max_disp_y = int(cc_max_displacement * ia_size_y)
    
    # Initialize arrays for results
    u = np.zeros((ia_n_y, ia_n_x))
    v = np.zeros((ia_n_y, ia_n_x))
    cc_peak = np.zeros((ia_n_y, ia_n_x))
    cc_peak_secondary = np.zeros((ia_n_y, ia_n_x))
    
    # Initialize counters for failed cross-correlations
    cc_failed_n = 0
    cc_subpx_failed_n = 0
    
    # Initialize array for cross-correlation peak image (if requested)
    return_cc_peak_im = 'return_cc_peak_im' in piv_params and piv_params['return_cc_peak_im']
    if return_cc_peak_im:
        cc_peak_im = np.zeros((ia_n_y * ia_size_y, ia_n_x * ia_size_x))
    else:
        cc_peak_im = None
    
    # Loop over interrogation areas
    for kx in range(ia_n_x):
        for ky in range(ia_n_y):
            # Skip if masked
            if status[ky, kx] & 1:
                continue
            
            # Get interrogation areas
            im_ia1 = ex_im1[ky*ia_size_y:(ky+1)*ia_size_y, kx*ia_size_x:(kx+1)*ia_size_x]
            im_ia2 = ex_im2[ky*ia_size_y:(ky+1)*ia_size_y, kx*ia_size_x:(kx+1)*ia_size_x]
            
            # Remove mean if requested
            if cc_remove_mean:
                im_ia1 = im_ia1 - np.mean(im_ia1)
                im_ia2 = im_ia2 - np.mean(im_ia2)
            
            # Apply window function
            im_ia1 = im_ia1 * window
            im_ia2 = im_ia2 * window
            
            # Compute cross-correlation
            if cc_method.lower() == 'fft':
                cc = cross_correlate_fft(im_ia1, im_ia2)
            elif cc_method.lower() == 'direct':
                max_disp = min(cc_max_disp_x, cc_max_disp_y)
                cc = cross_correlate_direct(im_ia1, im_ia2, max_disp)
            
            # Find the cross-correlation peak
            peak_x, peak_y, peak_val = find_peak_position(cc, cc_subpixel_method)
            
            # Store the peak value
            cc_peak[ky, kx] = peak_val
            
            # Check if peak is at the border of the correlation window
            if (peak_x <= 1 or peak_x >= cc.shape[1]-2 or 
                peak_y <= 1 or peak_y >= cc.shape[0]-2):
                # Mark as failed
                status[ky, kx] |= 2  # Set bit 1 (cross-correlation failed)
                cc_failed_n += 1
                continue
            
            # Check if peak is too far from the center
            center_x = cc.shape[1] // 2
            center_y = cc.shape[0] // 2
            if (abs(peak_x - center_x) > cc_max_disp_x or 
                abs(peak_y - center_y) > cc_max_disp_y):
                # Mark as failed
                status[ky, kx] |= 2  # Set bit 1 (cross-correlation failed)
                cc_failed_n += 1
                continue
            
            # Calculate displacement
            disp_x = peak_x - center_x
            disp_y = peak_y - center_y
            
            # Add initial shift
            u[ky, kx] = disp_x + ia_u0[ky, kx]
            v[ky, kx] = disp_y + ia_v0[ky, kx]
            
            # Find secondary peak (for signal-to-noise ratio)
            # Create a mask to remove the primary peak
            mask = np.ones_like(cc)
            mask_size = 5  # Size of the mask around the primary peak
            y_min = max(0, int(peak_y) - mask_size // 2)
            y_max = min(cc.shape[0], int(peak_y) + mask_size // 2 + 1)
            x_min = max(0, int(peak_x) - mask_size // 2)
            x_max = min(cc.shape[1], int(peak_x) + mask_size // 2 + 1)
            mask[y_min:y_max, x_min:x_max] = 0
            
            # Find the secondary peak
            cc_masked = cc * mask
            secondary_val = np.max(cc_masked)
            cc_peak_secondary[ky, kx] = secondary_val
            
            # Store cross-correlation peak image if requested
            if return_cc_peak_im:
                cc_peak_im[ky*ia_size_y:(ky+1)*ia_size_y, kx*ia_size_x:(kx+1)*ia_size_x] = cc
    
    # Store results in piv_data
    piv_data['u'] = u
    piv_data['v'] = v
    piv_data['status'] = status
    piv_data['cc_peak'] = cc_peak
    piv_data['cc_peak_secondary'] = cc_peak_secondary
    piv_data['cc_failed_n'] = cc_failed_n
    piv_data['cc_subpx_failed_n'] = cc_subpx_failed_n
    
    # Calculate signal-to-noise ratio
    piv_data['snr'] = cc_peak / cc_peak_secondary
    
    if return_cc_peak_im:
        return piv_data, cc_peak_im
    else:
        return piv_data
