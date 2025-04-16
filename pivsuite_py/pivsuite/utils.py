"""
Utility functions for PIVSuite
"""

import numpy as np
from scipy import ndimage
from skimage import io
import os
from typing import Tuple, List, Dict, Union, Optional, Any


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
        
    Returns
    -------
    np.ndarray
        Image as a numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = io.imread(image_path)
    
    # Convert to grayscale if RGB
    if len(img.shape) > 2:
        img = np.mean(img, axis=2).astype(np.float64)
    else:
        img = img.astype(np.float64)
    
    return img


def create_window_function(size_x: int, size_y: int, window_type: str = 'uniform') -> np.ndarray:
    """
    Create a window function for cross-correlation.
    
    Parameters
    ----------
    size_x : int
        Size of the window in x direction
    size_y : int
        Size of the window in y direction
    window_type : str
        Type of window function ('uniform', 'parzen', 'hanning', 'gaussian')
        
    Returns
    -------
    np.ndarray
        Window function as a 2D array
    """
    x = np.linspace(-1, 1, size_x)
    y = np.linspace(-1, 1, size_y)
    X, Y = np.meshgrid(x, y)
    
    if window_type.lower() == 'uniform':
        return np.ones((size_y, size_x))
    
    elif window_type.lower() == 'parzen':
        W = (1 - 2 * np.abs(X)) * (1 - 2 * np.abs(Y))
        return W
    
    elif window_type.lower() == 'hanning':
        W = (0.5 + 0.5 * np.cos(np.pi * X)) * (0.5 + 0.5 * np.cos(np.pi * Y))
        return W
    
    elif window_type.lower() == 'gaussian':
        sigma = 0.4
        W = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
        return W
    
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def cross_correlate_fft(window1: np.ndarray, window2: np.ndarray) -> np.ndarray:
    """
    Compute cross-correlation using FFT.
    
    Parameters
    ----------
    window1 : np.ndarray
        First window
    window2 : np.ndarray
        Second window
        
    Returns
    -------
    np.ndarray
        Cross-correlation function
    """
    # Remove mean
    win1 = window1 - np.mean(window1)
    win2 = window2 - np.mean(window2)
    
    # Compute standard deviation for normalization
    std1 = np.std(win1)
    std2 = np.std(win2)
    
    if std1 == 0 or std2 == 0:
        return np.zeros_like(win1)
    
    # Compute cross-correlation using FFT
    cc = np.fft.fftshift(np.real(np.fft.ifft2(
        np.conj(np.fft.fft2(win1)) * np.fft.fft2(win2)
    )))
    
    # Normalize
    cc = cc / (std1 * std2) / (win1.size)
    
    return cc


def cross_correlate_direct(window1: np.ndarray, window2: np.ndarray, max_displacement: int) -> np.ndarray:
    """
    Compute cross-correlation using direct convolution (for small windows).
    
    Parameters
    ----------
    window1 : np.ndarray
        First window
    window2 : np.ndarray
        Second window
    max_displacement : int
        Maximum displacement to consider
        
    Returns
    -------
    np.ndarray
        Cross-correlation function
    """
    # Remove mean
    win1 = window1 - np.mean(window1)
    win2 = window2 - np.mean(window2)
    
    # Compute standard deviation for normalization
    std1 = np.std(win1)
    std2 = np.std(win2)
    
    if std1 == 0 or std2 == 0:
        return np.zeros_like(win1)
    
    ny, nx = win1.shape
    cc = np.zeros((ny, nx))
    
    # Define center of cc
    if nx % 2 == 0:
        dx0 = nx // 2
    else:
        dx0 = nx // 2 + 0.5
        
    if ny % 2 == 0:
        dy0 = ny // 2
    else:
        dy0 = ny // 2 + 0.5
    
    dx0 = int(dx0)
    dy0 = int(dy0)
    
    # Pad windows
    pad = max_displacement
    win1_pad = np.zeros((ny + 2*pad, nx + 2*pad))
    win2_pad = np.zeros((ny + 2*pad, nx + 2*pad))
    win1_pad[pad:pad+ny, pad:pad+nx] = win1
    win2_pad[pad:pad+ny, pad:pad+nx] = win2
    
    # Compute cross-correlation by direct convolution
    for kx in range(-max_displacement, max_displacement+1):
        for ky in range(-max_displacement, max_displacement+1):
            if abs(kx) + abs(ky) > max_displacement:
                continue
                
            cc[dy0+ky, dx0+kx] = np.sum(
                win2_pad[ky+pad:ky+pad+ny, kx+pad:kx+pad+nx] * 
                win1_pad[pad:pad+ny, pad:pad+nx]
            )
    
    # Normalize
    cc = cc / (std1 * std2) / (win1.size)
    
    return cc


def find_peak_position(cc: np.ndarray, subpixel_method: str = 'gaussian') -> Tuple[float, float, float]:
    """
    Find the position of the peak in the cross-correlation function with subpixel accuracy.
    
    Parameters
    ----------
    cc : np.ndarray
        Cross-correlation function
    subpixel_method : str
        Method for subpixel interpolation ('gaussian', 'parabolic', 'centroid')
        
    Returns
    -------
    Tuple[float, float, float]
        (x, y, peak_value) - position of the peak and its value
    """
    # Find the peak position with pixel accuracy
    max_val = np.max(cc)
    max_idx = np.argmax(cc)
    max_y, max_x = np.unravel_index(max_idx, cc.shape)
    
    # Check if peak is at the border
    if (max_x == 0 or max_x == cc.shape[1]-1 or 
        max_y == 0 or max_y == cc.shape[0]-1):
        return max_x, max_y, max_val
    
    # Subpixel interpolation
    if subpixel_method.lower() == 'gaussian':
        # Fit a 2D Gaussian function to the peak
        log_c1 = np.log(cc[max_y, max_x-1])
        log_c2 = np.log(cc[max_y, max_x])
        log_c3 = np.log(cc[max_y, max_x+1])
        
        log_r1 = np.log(cc[max_y-1, max_x])
        log_r2 = np.log(cc[max_y, max_x])
        log_r3 = np.log(cc[max_y+1, max_x])
        
        # Compute peak position with subpixel accuracy
        if log_c1 + log_c3 - 2*log_c2 != 0 and log_r1 + log_r3 - 2*log_r2 != 0:
            dx = 0.5 * (log_c1 - log_c3) / (log_c1 + log_c3 - 2*log_c2)
            dy = 0.5 * (log_r1 - log_r3) / (log_r1 + log_r3 - 2*log_r2)
            
            # Compute peak value
            peak_val = np.exp(log_c2 - (log_c1 - log_c3)**2 / (8 * (log_c1 + log_c3 - 2*log_c2)) - 
                             (log_r1 - log_r3)**2 / (8 * (log_r1 + log_r3 - 2*log_r2)))
            
            return max_x + dx, max_y + dy, peak_val
    
    elif subpixel_method.lower() == 'parabolic':
        # Fit a parabola to the peak
        c1 = cc[max_y, max_x-1]
        c2 = cc[max_y, max_x]
        c3 = cc[max_y, max_x+1]
        
        r1 = cc[max_y-1, max_x]
        r2 = cc[max_y, max_x]
        r3 = cc[max_y+1, max_x]
        
        # Compute peak position with subpixel accuracy
        if c1 + c3 - 2*c2 != 0 and r1 + r3 - 2*r2 != 0:
            dx = 0.5 * (c1 - c3) / (c1 + c3 - 2*c2)
            dy = 0.5 * (r1 - r3) / (r1 + r3 - 2*r2)
            
            # Compute peak value
            peak_val = c2 - 0.25 * (c1 - c3) * dx
            
            return max_x + dx, max_y + dy, peak_val
    
    elif subpixel_method.lower() == 'centroid':
        # Use centroid method for subpixel accuracy
        # Extract a 3x3 region around the peak
        region = cc[max_y-1:max_y+2, max_x-1:max_x+2]
        
        # Compute centroid
        total = np.sum(region)
        if total > 0:
            y_coords, x_coords = np.mgrid[-1:2, -1:2]
            cx = np.sum(x_coords * region) / total
            cy = np.sum(y_coords * region) / total
            
            return max_x + cx, max_y + cy, max_val
    
    # Default: return pixel-accurate position
    return max_x, max_y, max_val


def validate_vectors(u: np.ndarray, v: np.ndarray, method: str = 'median', 
                    threshold: float = 2.0) -> np.ndarray:
    """
    Validate velocity vectors and mark spurious vectors.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    method : str
        Validation method ('median', 'mean', 'none')
    threshold : float
        Threshold for validation
        
    Returns
    -------
    np.ndarray
        Mask of valid vectors (1 = valid, 0 = invalid)
    """
    valid = np.ones_like(u, dtype=bool)
    
    if method.lower() == 'none':
        return valid
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    if method.lower() == 'median':
        # Compute median and median absolute deviation
        med_u = np.nanmedian(u)
        med_v = np.nanmedian(v)
        
        mad_u = np.nanmedian(np.abs(u - med_u))
        mad_v = np.nanmedian(np.abs(v - med_v))
        
        # Mark vectors as invalid if they deviate too much from the median
        valid = valid & (np.abs(u - med_u) < threshold * mad_u) & (np.abs(v - med_v) < threshold * mad_v)
    
    elif method.lower() == 'mean':
        # Compute mean and standard deviation
        mean_u = np.nanmean(u)
        mean_v = np.nanmean(v)
        
        std_u = np.nanstd(u)
        std_v = np.nanstd(v)
        
        # Mark vectors as invalid if they deviate too much from the mean
        valid = valid & (np.abs(u - mean_u) < threshold * std_u) & (np.abs(v - mean_v) < threshold * std_v)
    
    return valid


def replace_invalid_vectors(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, 
                           valid: np.ndarray, method: str = 'interpolate') -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace invalid vectors.
    
    Parameters
    ----------
    x : np.ndarray
        x-coordinates of vectors
    y : np.ndarray
        y-coordinates of vectors
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    valid : np.ndarray
        Mask of valid vectors (1 = valid, 0 = invalid)
    method : str
        Replacement method ('interpolate', 'mean', 'none')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (u_replaced, v_replaced) - velocity field with replaced vectors
    """
    u_replaced = u.copy()
    v_replaced = v.copy()
    
    if method.lower() == 'none':
        return u_replaced, v_replaced
    
    # Mark invalid vectors as NaN
    u_replaced[~valid] = np.nan
    v_replaced[~valid] = np.nan
    
    if method.lower() == 'interpolate':
        # Use scipy's griddata for interpolation
        from scipy.interpolate import griddata
        
        # Get coordinates of valid vectors
        x_valid = x[valid]
        y_valid = y[valid]
        u_valid = u[valid]
        v_valid = v[valid]
        
        # Get coordinates of invalid vectors
        x_invalid = x[~valid]
        y_invalid = y[~valid]
        
        if len(x_valid) > 3:  # Need at least 3 points for interpolation
            # Interpolate u and v at invalid positions
            u_interp = griddata((x_valid, y_valid), u_valid, (x_invalid, y_invalid), method='linear')
            v_interp = griddata((x_valid, y_valid), v_valid, (x_invalid, y_invalid), method='linear')
            
            # Replace invalid vectors with interpolated values
            u_replaced[~valid] = u_interp
            v_replaced[~valid] = v_interp
    
    elif method.lower() == 'mean':
        # Replace with mean of valid vectors
        u_mean = np.nanmean(u_replaced)
        v_mean = np.nanmean(v_replaced)
        
        u_replaced[~valid] = u_mean
        v_replaced[~valid] = v_mean
    
    # Handle any remaining NaNs
    u_replaced = np.nan_to_num(u_replaced)
    v_replaced = np.nan_to_num(v_replaced)
    
    return u_replaced, v_replaced


def smooth_vector_field(u: np.ndarray, v: np.ndarray, method: str = 'gaussian', 
                       sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth the velocity field.
    
    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    method : str
        Smoothing method ('gaussian', 'median', 'none')
    sigma : float
        Smoothing parameter
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (u_smoothed, v_smoothed) - smoothed velocity field
    """
    if method.lower() == 'none':
        return u, v
    
    u_smoothed = u.copy()
    v_smoothed = v.copy()
    
    if method.lower() == 'gaussian':
        # Apply Gaussian filter
        u_smoothed = ndimage.gaussian_filter(u, sigma=sigma)
        v_smoothed = ndimage.gaussian_filter(v, sigma=sigma)
    
    elif method.lower() == 'median':
        # Apply median filter
        size = int(2 * sigma + 1)
        u_smoothed = ndimage.median_filter(u, size=size)
        v_smoothed = ndimage.median_filter(v, size=size)
    
    return u_smoothed, v_smoothed


def compute_vorticity(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute vorticity from velocity field.
    
    Parameters
    ----------
    x : np.ndarray
        x-coordinates of vectors
    y : np.ndarray
        y-coordinates of vectors
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
        
    Returns
    -------
    np.ndarray
        Vorticity field
    """
    # Compute grid spacing
    dx = np.mean(np.diff(x[0, :]))
    dy = np.mean(np.diff(y[:, 0]))
    
    # Compute derivatives
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    
    # Compute vorticity
    vorticity = dv_dx - du_dy
    
    return vorticity
