"""
Cross-correlation module for PIVSuite Python

This module handles the cross-correlation between interrogation areas.
It corresponds to the pivCrossCorr.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import numba

from ..utils.math import std_fast, create_window_function


def cross_correlate(
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
    cc_remove_ia_mean = piv_params.get('cc_remove_ia_mean', 1.0)
    cc_max_displacement = piv_params.get('cc_max_displacement', 0.7)
    cc_subpixel_method = piv_params.get('cc_subpixel_method', 'gaussian')
    cc_correct_window_bias = piv_params.get('cc_correct_window_bias', True)
    
    # Get dimensions
    ia_n_y, ia_n_x = piv_data['status'].shape
    
    # Get status array
    status = piv_data['status'].copy()
    
    # Get initial shift of interrogation areas
    ia_u0 = piv_data.get('ia_u0', np.zeros((ia_n_y, ia_n_x)))
    ia_v0 = piv_data.get('ia_v0', np.zeros((ia_n_y, ia_n_x)))
    
    # Create window function W and loss-of-correlation function F
    W, F = create_window_function(ia_size_x, ia_size_y, cc_window)
    
    # Limit F to not be too small
    if F is not None:
        F[F < 0.5] = 0.5
    
    # Peak position is shifted by 1 or 0.5 px, depending on IA size
    if ia_size_x % 2 == 0:
        cc_px_shift_x = 1
    else:
        cc_px_shift_x = 0.5
    
    if ia_size_y % 2 == 0:
        cc_px_shift_y = 1
    else:
        cc_px_shift_y = 0.5
    
    # Initialize arrays for results
    u = np.zeros((ia_n_y, ia_n_x))
    v = np.zeros((ia_n_y, ia_n_x))
    cc_peak = np.zeros((ia_n_y, ia_n_x))
    cc_peak_secondary = np.zeros((ia_n_y, ia_n_x))
    cc_std1 = np.full((ia_n_y, ia_n_x), np.nan)
    cc_std2 = np.full((ia_n_y, ia_n_x), np.nan)
    cc_mean1 = np.full((ia_n_y, ia_n_x), np.nan)
    cc_mean2 = np.full((ia_n_y, ia_n_x), np.nan)
    
    # Initialize counters for failed cross-correlations
    cc_failed_n = 0
    cc_subpx_failed_n = 0
    
    # Initialize array for cross-correlation peak image (if requested)
    return_cc_peak_im = piv_params.get('return_cc_peak_im', False)
    if return_cc_peak_im:
        cc_peak_im = np.full_like(ex_im1, np.nan)
    else:
        cc_peak_im = None
    
    # Loop over interrogation areas
    for kx in range(ia_n_x):
        for ky in range(ia_n_y):
            # Skip if masked
            fail_flag = status[ky, kx]
            if fail_flag & 1:
                cc = np.full((ia_size_y, ia_size_x), np.nan)
                aux_peak = np.nan
                aux_std1 = np.nan
                aux_std2 = np.nan
                aux_mean1 = np.nan
                aux_mean2 = np.nan
                u_px = ia_size_x // 2
                v_px = ia_size_y // 2
            else:
                # Get interrogation areas
                im_ia1 = ex_im1[ky*ia_size_y:(ky+1)*ia_size_y, kx*ia_size_x:(kx+1)*ia_size_x]
                im_ia2 = ex_im2[ky*ia_size_y:(ky+1)*ia_size_y, kx*ia_size_x:(kx+1)*ia_size_x]
                
                # Remove IA mean
                aux_mean1 = np.mean(im_ia1)
                aux_mean2 = np.mean(im_ia2)
                im_ia1 = im_ia1 - cc_remove_ia_mean * aux_mean1
                im_ia2 = im_ia2 - cc_remove_ia_mean * aux_mean2
                
                # Apply windowing function
                im_ia1 = im_ia1 * W
                im_ia2 = im_ia2 * W
                
                # Compute rms for normalization of cross-correlation
                aux_std1 = std_fast(im_ia1)
                aux_std2 = std_fast(im_ia2)
                
                # Compute cross-correlation
                if cc_method.lower() == 'fft':
                    cc = fft_cross_correlate(im_ia1, im_ia2, aux_std1, aux_std2)
                    
                    # Find the cross-correlation peak
                    aux_peak = np.max(cc)
                    u_px = np.argmax(np.max(cc, axis=0))
                    v_px = np.argmax(cc[:, u_px])
                    
                elif cc_method.lower() == 'dcn':
                    max_disp = min(int(cc_max_displacement * ia_size_x), int(cc_max_displacement * ia_size_y))
                    cc = dcn_cross_correlate(im_ia1, im_ia2, max_disp, aux_std1, aux_std2)
                    
                    # Find the cross-correlation peak
                    aux_peak = np.max(cc)
                    u_px = np.argmax(np.max(cc, axis=0))
                    v_px = np.argmax(cc[:, u_px])
                    
                    # If peak is not at the center, use FFT method
                    if (u_px != ia_size_x//2 + cc_px_shift_x) or (v_px != ia_size_y//2 + cc_px_shift_y):
                        cc = fft_cross_correlate(im_ia1, im_ia2, aux_std1, aux_std2)
                        aux_peak = np.max(cc)
                        u_px = np.argmax(np.max(cc, axis=0))
                        v_px = np.argmax(cc[:, u_px])
                
                # Check if the displacement is too large
                if (abs(u_px - ia_size_x//2 - cc_px_shift_x) > cc_max_displacement * ia_size_x) or \
                   (abs(v_px - ia_size_y//2 - cc_px_shift_y) > cc_max_displacement * ia_size_y):
                    fail_flag |= 2  # Set bit 1 (cross-correlation failed)
                    cc_failed_n += 1
                
                # Correct cc peak for bias caused by interrogation window
                if cc_correct_window_bias and F is not None:
                    cc_cor = cc / F
                else:
                    cc_cor = cc
                
                # Sub-pixel interpolation (2x3point Gaussian fit)
                try:
                    # Check if peak is at the border
                    if (u_px <= 0 or u_px >= cc.shape[1]-1 or 
                        v_px <= 0 or v_px >= cc.shape[0]-1):
                        raise ValueError("Peak at border")
                    
                    # Compute sub-pixel displacement
                    du = (np.log(cc_cor[v_px, u_px-1]) - np.log(cc_cor[v_px, u_px+1])) / \
                         (np.log(cc_cor[v_px, u_px-1]) + np.log(cc_cor[v_px, u_px+1]) - 2*np.log(cc_cor[v_px, u_px])) / 2
                    
                    dv = (np.log(cc_cor[v_px-1, u_px]) - np.log(cc_cor[v_px+1, u_px])) / \
                         (np.log(cc_cor[v_px-1, u_px]) + np.log(cc_cor[v_px+1, u_px]) - 2*np.log(cc_cor[v_px, u_px])) / 2
                    
                    # Check if the result is valid
                    if np.isnan(du) or np.isnan(dv) or np.isinf(du) or np.isinf(dv):
                        raise ValueError("Invalid sub-pixel displacement")
                    
                except Exception:
                    fail_flag |= 4  # Set bit 2 (peak detection failed)
                    cc_subpx_failed_n += 1
                    du = 0
                    dv = 0
            
            # Save the results
            if fail_flag == 0:
                u[ky, kx] = ia_u0[ky, kx] + u_px + du - ia_size_x//2 - cc_px_shift_x
                v[ky, kx] = ia_v0[ky, kx] + v_px + dv - ia_size_y//2 - cc_px_shift_y
            else:
                u[ky, kx] = np.nan
                v[ky, kx] = np.nan
            
            status[ky, kx] = fail_flag
            
            if return_cc_peak_im:
                cc_peak_im[ky*ia_size_y:(ky+1)*ia_size_y, kx*ia_size_x:(kx+1)*ia_size_x] = cc
            
            cc_peak[ky, kx] = aux_peak
            cc_std1[ky, kx] = aux_std1
            cc_std2[ky, kx] = aux_std2
            cc_mean1[ky, kx] = aux_mean1
            cc_mean2[ky, kx] = aux_mean2
            
            # Find secondary peak
            try:
                cc_copy = cc.copy()
                # Create a mask to remove the primary peak
                mask_size = 5  # Size of the mask around the primary peak
                y_min = max(0, v_px - mask_size // 2)
                y_max = min(cc.shape[0], v_px + mask_size // 2 + 1)
                x_min = max(0, u_px - mask_size // 2)
                x_max = min(cc.shape[1], u_px + mask_size // 2 + 1)
                cc_copy[y_min:y_max, x_min:x_max] = 0
                
                cc_peak_secondary[ky, kx] = np.max(cc_copy)
            except Exception:
                try:
                    cc_copy = cc.copy()
                    # Try with a smaller mask
                    mask_size = 3
                    y_min = max(0, v_px - mask_size // 2)
                    y_max = min(cc.shape[0], v_px + mask_size // 2 + 1)
                    x_min = max(0, u_px - mask_size // 2)
                    x_max = min(cc.shape[1], u_px + mask_size // 2 + 1)
                    cc_copy[y_min:y_max, x_min:x_max] = 0
                    
                    cc_peak_secondary[ky, kx] = np.max(cc_copy)
                except Exception:
                    cc_peak_secondary[ky, kx] = np.nan
    
    # Get IAs where CC failed, and coordinates of corresponding IAs
    cc_failed_i = np.bitwise_and(status, 2).astype(bool)
    cc_subpx_failed_i = np.bitwise_and(status, 4).astype(bool)
    
    # Store results in piv_data
    piv_data['status'] = status
    piv_data['u'] = u
    piv_data['v'] = v
    piv_data['cc_peak'] = cc_peak
    piv_data['cc_peak_secondary'] = cc_peak_secondary
    piv_data['cc_std1'] = cc_std1
    piv_data['cc_std2'] = cc_std2
    piv_data['cc_mean1'] = cc_mean1
    piv_data['cc_mean2'] = cc_mean2
    piv_data['cc_failed_n'] = cc_failed_n
    piv_data['cc_subpx_failed_n'] = cc_subpx_failed_n
    piv_data['cc_w'] = W
    
    # Remove fields that are no longer needed
    if 'ia_u0' in piv_data:
        del piv_data['ia_u0']
    if 'ia_v0' in piv_data:
        del piv_data['ia_v0']
    
    if return_cc_peak_im:
        return piv_data, cc_peak_im
    else:
        return piv_data


def fft_cross_correlate(im1: np.ndarray, im2: np.ndarray, std1: float, std2: float) -> np.ndarray:
    """
    Compute cross-correlation using FFT.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    std1 : float
        Standard deviation of first image
    std2 : float
        Standard deviation of second image
        
    Returns
    -------
    np.ndarray
        Cross-correlation function
    """
    # Compute cross-correlation using FFT
    cc = np.fft.fftshift(np.real(np.fft.ifft2(
        np.conj(np.fft.fft2(im1)) * np.fft.fft2(im2)
    )))
    
    # Normalize
    cc = cc / (std1 * std2) / (im1.size)
    
    return cc


def dcn_cross_correlate(im1: np.ndarray, im2: np.ndarray, max_disp: int, std1: float, std2: float) -> np.ndarray:
    """
    Compute cross-correlation using direct convolution.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    max_disp : int
        Maximum displacement to consider
    std1 : float
        Standard deviation of first image
    std2 : float
        Standard deviation of second image
        
    Returns
    -------
    np.ndarray
        Cross-correlation function
    """
    nx = im1.shape[1]
    ny = im1.shape[0]
    cc = np.zeros((ny, nx))
    
    # Create variables defining where is cc(0,0)
    dx0 = nx // 2
    dy0 = ny // 2
    if nx % 2 == 0:
        dx0 += 1
    else:
        dx0 += 0.5
    
    if ny % 2 == 0:
        dy0 += 1
    else:
        dy0 += 0.5
    
    dx0 = int(dx0)
    dy0 = int(dy0)
    
    # Pad IAs
    im1_pad = np.zeros((ny + 2*max_disp, nx + 2*max_disp))
    im2_pad = np.zeros((ny + 2*max_disp, nx + 2*max_disp))
    im1_pad[max_disp:max_disp+ny, max_disp:max_disp+nx] = im1
    im2_pad[max_disp:max_disp+ny, max_disp:max_disp+nx] = im2
    
    # Convolve
    for kx in range(-max_disp, max_disp+1):
        for ky in range(-max_disp, max_disp+1):
            if abs(kx) + abs(ky) > max_disp:
                continue
            
            cc[dy0+ky, dx0+kx] = np.sum(
                im2_pad[ky+max_disp:ky+max_disp+ny, kx+max_disp:kx+max_disp+nx] * 
                im1_pad[max_disp:max_disp+ny, max_disp:max_disp+nx]
            )
    
    # Normalize
    cc = cc / (std1 * std2) / (im1.size)
    
    return cc
