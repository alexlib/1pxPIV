"""
Interrogation module for PIVSuite

This module handles the creation and manipulation of interrogation areas.
It corresponds to the pivInterrogate.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Tuple, Dict, Union, Optional, Any
from scipy.interpolate import griddata, RegularGridInterpolator
import numba

from .utils import create_window_function


def interrogate_images(
    im1: np.ndarray, 
    im2: np.ndarray, 
    piv_data: Dict[str, Any], 
    piv_params: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Split images into interrogation areas and create expanded images suitable for cross-correlation.
    
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
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
        ex_im1: Expanded first image
        ex_im2: Expanded second image
        piv_data: Updated PIV results
    """
    # Get image dimensions
    im_size_y, im_size_x = im1.shape
    
    # Get interrogation area parameters
    ia_size_x = piv_params.get('ia_size_x', 32)
    ia_size_y = piv_params.get('ia_size_y', 32)
    ia_step_x = piv_params.get('ia_step_x', 16)
    ia_step_y = piv_params.get('ia_step_y', 16)
    ia_method = piv_params.get('ia_method', 'basic')
    
    # Calculate number of interrogation areas
    ia_n_x = int((im_size_x - ia_size_x) / ia_step_x) + 1
    ia_n_y = int((im_size_y - ia_size_y) / ia_step_y) + 1
    
    # Create grid of interrogation area centers
    x = np.arange(ia_size_x/2, ia_size_x/2 + ia_n_x*ia_step_x, ia_step_x)
    y = np.arange(ia_size_y/2, ia_size_y/2 + ia_n_y*ia_step_y, ia_step_y)
    X, Y = np.meshgrid(x, y)
    
    # Initialize expanded images
    ex_im1 = np.zeros((ia_n_y * ia_size_y, ia_n_x * ia_size_x))
    ex_im2 = np.zeros((ia_n_y * ia_size_y, ia_n_x * ia_size_x))
    
    # Initialize status array (0 = valid, 1 = masked)
    status = np.zeros((ia_n_y, ia_n_x), dtype=np.uint16)
    
    # Get mask if available
    mask1 = piv_params.get('im_mask1', None)
    mask2 = piv_params.get('im_mask2', None)
    
    # Apply masks if available
    if mask1 is not None:
        if isinstance(mask1, str):
            # Load mask from file
            from skimage import io
            mask1 = io.imread(mask1) > 0
        im1 = np.where(mask1, np.nan, im1)
    
    if mask2 is not None:
        if isinstance(mask2, str):
            # Load mask from file
            from skimage import io
            mask2 = io.imread(mask2) > 0
        im2 = np.where(mask2, np.nan, im2)
    
    # Process interrogation areas based on method
    if ia_method.lower() == 'basic':
        # Standard interrogation - no IA offset or deformation
        ex_im1, ex_im2, status = process_basic_interrogation(
            im1, im2, ia_n_x, ia_n_y, ia_size_x, ia_size_y, ex_im1, ex_im2, status
        )
        
        # Set interpolated velocity to zeros
        u0 = np.zeros((ia_n_y, ia_n_x))
        v0 = np.zeros((ia_n_y, ia_n_x))
        
    elif ia_method.lower() == 'offset':
        # Interrogation with offset of IA, no deformation of IA
        # Get previous velocity field if available
        if 'u' in piv_data and 'v' in piv_data and 'x' in piv_data and 'y' in piv_data:
            # Interpolate the velocity estimates to the new grid
            u_est = piv_data['u']
            v_est = piv_data['v']
            x_est = piv_data['x']
            y_est = piv_data['y']
            
            # Interpolate to new grid
            u0 = interpolate_velocity(x_est, y_est, u_est, X, Y)
            v0 = interpolate_velocity(x_est, y_est, v_est, X, Y)
        else:
            # No previous velocity field, use zeros
            u0 = np.zeros((ia_n_y, ia_n_x))
            v0 = np.zeros((ia_n_y, ia_n_x))
        
        # Process interrogation areas with offset
        ex_im1, ex_im2, status = process_offset_interrogation(
            im1, im2, ia_n_x, ia_n_y, ia_size_x, ia_size_y, 
            ex_im1, ex_im2, status, u0, v0
        )
        
    elif ia_method.lower() == 'deform':
        # Interrogation with deformation of IA
        # Get previous velocity field if available
        if 'u' in piv_data and 'v' in piv_data and 'x' in piv_data and 'y' in piv_data:
            # Interpolate the velocity estimates to the new grid
            u_est = piv_data['u']
            v_est = piv_data['v']
            x_est = piv_data['x']
            y_est = piv_data['y']
            
            # Interpolate to new grid
            u0 = interpolate_velocity(x_est, y_est, u_est, X, Y)
            v0 = interpolate_velocity(x_est, y_est, v_est, X, Y)
        else:
            # No previous velocity field, use zeros
            u0 = np.zeros((ia_n_y, ia_n_x))
            v0 = np.zeros((ia_n_y, ia_n_x))
        
        # Process interrogation areas with deformation
        ex_im1, ex_im2, status = process_deform_interrogation(
            im1, im2, ia_n_x, ia_n_y, ia_size_x, ia_size_y, 
            ex_im1, ex_im2, status, u0, v0, X, Y
        )
    
    # Store results in piv_data
    piv_data['x'] = X
    piv_data['y'] = Y
    piv_data['u'] = np.full_like(X, np.nan)
    piv_data['v'] = np.full_like(X, np.nan)
    piv_data['n'] = X.size
    piv_data['status'] = status
    piv_data['masked_n'] = np.sum(status & 1)
    piv_data['im_size_x'] = im_size_x
    piv_data['im_size_y'] = im_size_y
    piv_data['ia_size_x'] = ia_size_x
    piv_data['ia_size_y'] = ia_size_y
    piv_data['ia_step_x'] = ia_step_x
    piv_data['ia_step_y'] = ia_step_y
    piv_data['ia_u0'] = u0
    piv_data['ia_v0'] = v0
    
    return ex_im1, ex_im2, piv_data


@numba.njit
def process_basic_interrogation(
    im1: np.ndarray, 
    im2: np.ndarray, 
    ia_n_x: int, 
    ia_n_y: int, 
    ia_size_x: int, 
    ia_size_y: int, 
    ex_im1: np.ndarray, 
    ex_im2: np.ndarray, 
    status: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process interrogation areas using basic method (no offset or deformation).
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    ia_n_x : int
        Number of interrogation areas in x direction
    ia_n_y : int
        Number of interrogation areas in y direction
    ia_size_x : int
        Size of interrogation area in x direction
    ia_size_y : int
        Size of interrogation area in y direction
    ex_im1 : np.ndarray
        Expanded first image (output)
    ex_im2 : np.ndarray
        Expanded second image (output)
    status : np.ndarray
        Status array (output)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ex_im1: Expanded first image
        ex_im2: Expanded second image
        status: Updated status array
    """
    for kx in range(ia_n_x):
        for ky in range(ia_n_y):
            # Get the interrogation areas
            ia_start_x = kx * ia_size_x
            ia_stop_x = (kx + 1) * ia_size_x
            ia_start_y = ky * ia_size_y
            ia_stop_y = (ky + 1) * ia_size_y
            
            im_start_x = kx * ia_size_x
            im_stop_x = (kx + 1) * ia_size_x
            im_start_y = ky * ia_size_y
            im_stop_y = (ky + 1) * ia_size_y
            
            # Extract interrogation areas
            im_ia1 = im1[im_start_y:im_stop_y, im_start_x:im_stop_x].copy()
            im_ia2 = im2[im_start_y:im_stop_y, im_start_x:im_stop_x].copy()
            
            # Check for NaN values (masked pixels)
            masked1 = np.isnan(im_ia1)
            masked2 = np.isnan(im_ia2)
            
            # Replace NaN values with mean of non-NaN values
            if np.any(masked1):
                valid_pixels = ~masked1
                if np.any(valid_pixels):
                    mean_val = np.mean(im_ia1[valid_pixels])
                    im_ia1[masked1] = mean_val
            
            if np.any(masked2):
                valid_pixels = ~masked2
                if np.any(valid_pixels):
                    mean_val = np.mean(im_ia2[valid_pixels])
                    im_ia2[masked2] = mean_val
            
            # Copy to expanded images
            ex_im1[ia_start_y:ia_stop_y, ia_start_x:ia_stop_x] = im_ia1
            ex_im2[ia_start_y:ia_stop_y, ia_start_x:ia_stop_x] = im_ia2
            
            # Check if too many pixels are masked
            if np.sum(masked1 | masked2) > 0.5 * ia_size_x * ia_size_y:
                status[ky, kx] = 1  # Mark as masked
    
    return ex_im1, ex_im2, status


def process_offset_interrogation(
    im1: np.ndarray, 
    im2: np.ndarray, 
    ia_n_x: int, 
    ia_n_y: int, 
    ia_size_x: int, 
    ia_size_y: int, 
    ex_im1: np.ndarray, 
    ex_im2: np.ndarray, 
    status: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process interrogation areas using offset method.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    ia_n_x : int
        Number of interrogation areas in x direction
    ia_n_y : int
        Number of interrogation areas in y direction
    ia_size_x : int
        Size of interrogation area in x direction
    ia_size_y : int
        Size of interrogation area in y direction
    ex_im1 : np.ndarray
        Expanded first image (output)
    ex_im2 : np.ndarray
        Expanded second image (output)
    status : np.ndarray
        Status array (output)
    u0 : np.ndarray
        x-component of velocity field
    v0 : np.ndarray
        y-component of velocity field
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ex_im1: Expanded first image
        ex_im2: Expanded second image
        status: Updated status array
    """
    im_size_y, im_size_x = im1.shape
    
    for kx in range(ia_n_x):
        for ky in range(ia_n_y):
            # Get the interrogation areas
            ia_start_x = kx * ia_size_x
            ia_stop_x = (kx + 1) * ia_size_x
            ia_start_y = ky * ia_size_y
            ia_stop_y = (ky + 1) * ia_size_y
            
            # Get the offset for the second image
            offset_x = int(round(u0[ky, kx]))
            offset_y = int(round(v0[ky, kx]))
            
            # Calculate coordinates for the first image
            im1_start_x = kx * ia_size_x
            im1_stop_x = (kx + 1) * ia_size_x
            im1_start_y = ky * ia_size_y
            im1_stop_y = (ky + 1) * ia_size_y
            
            # Calculate coordinates for the second image (with offset)
            im2_start_x = im1_start_x + offset_x
            im2_stop_x = im1_stop_x + offset_x
            im2_start_y = im1_start_y + offset_y
            im2_stop_y = im1_stop_y + offset_y
            
            # Check if the second image coordinates are within bounds
            if (im2_start_x < 0 or im2_stop_x > im_size_x or 
                im2_start_y < 0 or im2_stop_y > im_size_y):
                # Mark as masked if out of bounds
                status[ky, kx] = 1
                continue
            
            # Extract interrogation areas
            im_ia1 = im1[im1_start_y:im1_stop_y, im1_start_x:im1_stop_x].copy()
            im_ia2 = im2[im2_start_y:im2_stop_y, im2_start_x:im2_stop_x].copy()
            
            # Check for NaN values (masked pixels)
            masked1 = np.isnan(im_ia1)
            masked2 = np.isnan(im_ia2)
            
            # Replace NaN values with mean of non-NaN values
            if np.any(masked1):
                valid_pixels = ~masked1
                if np.any(valid_pixels):
                    mean_val = np.mean(im_ia1[valid_pixels])
                    im_ia1[masked1] = mean_val
            
            if np.any(masked2):
                valid_pixels = ~masked2
                if np.any(valid_pixels):
                    mean_val = np.mean(im_ia2[valid_pixels])
                    im_ia2[masked2] = mean_val
            
            # Copy to expanded images
            ex_im1[ia_start_y:ia_stop_y, ia_start_x:ia_stop_x] = im_ia1
            ex_im2[ia_start_y:ia_stop_y, ia_start_x:ia_stop_x] = im_ia2
            
            # Check if too many pixels are masked
            if np.sum(masked1 | masked2) > 0.5 * ia_size_x * ia_size_y:
                status[ky, kx] = 1  # Mark as masked
    
    return ex_im1, ex_im2, status


def process_deform_interrogation(
    im1: np.ndarray, 
    im2: np.ndarray, 
    ia_n_x: int, 
    ia_n_y: int, 
    ia_size_x: int, 
    ia_size_y: int, 
    ex_im1: np.ndarray, 
    ex_im2: np.ndarray, 
    status: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process interrogation areas using deformation method.
    
    Parameters
    ----------
    im1 : np.ndarray
        First image
    im2 : np.ndarray
        Second image
    ia_n_x : int
        Number of interrogation areas in x direction
    ia_n_y : int
        Number of interrogation areas in y direction
    ia_size_x : int
        Size of interrogation area in x direction
    ia_size_y : int
        Size of interrogation area in y direction
    ex_im1 : np.ndarray
        Expanded first image (output)
    ex_im2 : np.ndarray
        Expanded second image (output)
    status : np.ndarray
        Status array (output)
    u0 : np.ndarray
        x-component of velocity field
    v0 : np.ndarray
        y-component of velocity field
    X : np.ndarray
        x-coordinates of interrogation area centers
    Y : np.ndarray
        y-coordinates of interrogation area centers
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ex_im1: Expanded first image
        ex_im2: Expanded second image
        status: Updated status array
    """
    im_size_y, im_size_x = im1.shape
    
    # Create interpolation function for velocity field
    points = np.column_stack((Y.flatten(), X.flatten()))
    u_values = u0.flatten()
    v_values = v0.flatten()
    
    # Create grid for the entire image
    y_grid, x_grid = np.mgrid[0:im_size_y, 0:im_size_x]
    
    # Interpolate velocity field to the entire image
    u_interp = griddata(points, u_values, (y_grid, x_grid), method='linear', fill_value=0)
    v_interp = griddata(points, v_values, (y_grid, x_grid), method='linear', fill_value=0)
    
    # Create deformed coordinates for the second image
    y_deform = y_grid - 0.5 * v_interp
    x_deform = x_grid - 0.5 * u_interp
    
    # Create deformed coordinates for the first image
    y_deform1 = y_grid + 0.5 * v_interp
    x_deform1 = x_grid + 0.5 * u_interp
    
    # Ensure coordinates are within bounds
    y_deform = np.clip(y_deform, 0, im_size_y - 1)
    x_deform = np.clip(x_deform, 0, im_size_x - 1)
    y_deform1 = np.clip(y_deform1, 0, im_size_y - 1)
    x_deform1 = np.clip(x_deform1, 0, im_size_x - 1)
    
    # Create interpolation functions for the images
    from scipy.interpolate import RegularGridInterpolator
    
    # Create regular grid interpolators
    y_coords = np.arange(im_size_y)
    x_coords = np.arange(im_size_x)
    
    # Handle NaN values in images
    im1_interp = im1.copy()
    im2_interp = im2.copy()
    
    # Replace NaN values with mean
    if np.any(np.isnan(im1_interp)):
        mean_val = np.nanmean(im1_interp)
        im1_interp[np.isnan(im1_interp)] = mean_val
    
    if np.any(np.isnan(im2_interp)):
        mean_val = np.nanmean(im2_interp)
        im2_interp[np.isnan(im2_interp)] = mean_val
    
    interp_func1 = RegularGridInterpolator((y_coords, x_coords), im1_interp, 
                                          bounds_error=False, fill_value=0)
    interp_func2 = RegularGridInterpolator((y_coords, x_coords), im2_interp, 
                                          bounds_error=False, fill_value=0)
    
    # Interpolate deformed images
    points_to_interp1 = np.column_stack((y_deform1.flatten(), x_deform1.flatten()))
    points_to_interp2 = np.column_stack((y_deform.flatten(), x_deform.flatten()))
    
    im1_deformed = interp_func1(points_to_interp1).reshape(im_size_y, im_size_x)
    im2_deformed = interp_func2(points_to_interp2).reshape(im_size_y, im_size_x)
    
    # Process interrogation areas from deformed images
    for kx in range(ia_n_x):
        for ky in range(ia_n_y):
            # Get the interrogation areas
            ia_start_x = kx * ia_size_x
            ia_stop_x = (kx + 1) * ia_size_x
            ia_start_y = ky * ia_size_y
            ia_stop_y = (ky + 1) * ia_size_y
            
            # Calculate coordinates for the images
            im_start_x = kx * ia_size_x
            im_stop_x = (kx + 1) * ia_size_x
            im_start_y = ky * ia_size_y
            im_stop_y = (ky + 1) * ia_size_y
            
            # Check if coordinates are within bounds
            if (im_start_x < 0 or im_stop_x > im_size_x or 
                im_start_y < 0 or im_stop_y > im_size_y):
                # Mark as masked if out of bounds
                status[ky, kx] = 1
                continue
            
            # Extract interrogation areas from deformed images
            im_ia1 = im1_deformed[im_start_y:im_stop_y, im_start_x:im_stop_x].copy()
            im_ia2 = im2_deformed[im_start_y:im_stop_y, im_start_x:im_stop_x].copy()
            
            # Check for NaN values (masked pixels)
            masked1 = np.isnan(im_ia1)
            masked2 = np.isnan(im_ia2)
            
            # Replace NaN values with mean of non-NaN values
            if np.any(masked1):
                valid_pixels = ~masked1
                if np.any(valid_pixels):
                    mean_val = np.mean(im_ia1[valid_pixels])
                    im_ia1[masked1] = mean_val
            
            if np.any(masked2):
                valid_pixels = ~masked2
                if np.any(valid_pixels):
                    mean_val = np.mean(im_ia2[valid_pixels])
                    im_ia2[masked2] = mean_val
            
            # Copy to expanded images
            ex_im1[ia_start_y:ia_stop_y, ia_start_x:ia_stop_x] = im_ia1
            ex_im2[ia_start_y:ia_stop_y, ia_start_x:ia_stop_x] = im_ia2
            
            # Check if too many pixels are masked
            if np.sum(masked1 | masked2) > 0.5 * ia_size_x * ia_size_y:
                status[ky, kx] = 1  # Mark as masked
    
    return ex_im1, ex_im2, status


def interpolate_velocity(
    x_old: np.ndarray, 
    y_old: np.ndarray, 
    v_old: np.ndarray, 
    x_new: np.ndarray, 
    y_new: np.ndarray
) -> np.ndarray:
    """
    Interpolate velocity field to a new grid.
    
    Parameters
    ----------
    x_old : np.ndarray
        x-coordinates of old grid
    y_old : np.ndarray
        y-coordinates of old grid
    v_old : np.ndarray
        Velocity field on old grid
    x_new : np.ndarray
        x-coordinates of new grid
    y_new : np.ndarray
        y-coordinates of new grid
        
    Returns
    -------
    np.ndarray
        Interpolated velocity field on new grid
    """
    # Handle NaN values in the velocity field
    v_old_clean = v_old.copy()
    nan_mask = np.isnan(v_old_clean)
    
    if np.any(nan_mask):
        # Replace NaN values with interpolated values
        points = np.column_stack((y_old[~nan_mask].flatten(), x_old[~nan_mask].flatten()))
        values = v_old_clean[~nan_mask].flatten()
        
        if len(points) > 3:  # Need at least 3 points for interpolation
            grid_y, grid_x = np.mgrid[
                np.min(y_old):np.max(y_old):complex(0, y_old.shape[0]),
                np.min(x_old):np.max(x_old):complex(0, x_old.shape[1])
            ]
            
            v_interp = griddata(points, values, (grid_y, grid_x), method='linear')
            v_old_clean = v_interp
        else:
            # Not enough points for interpolation, use mean
            v_old_clean[nan_mask] = np.nanmean(v_old_clean)
    
    # Interpolate to new grid
    points = np.column_stack((y_old.flatten(), x_old.flatten()))
    values = v_old_clean.flatten()
    
    v_new = griddata(points, values, (y_new, x_new), method='linear')
    
    # Handle any remaining NaN values
    if np.any(np.isnan(v_new)):
        v_new[np.isnan(v_new)] = np.nanmean(v_new)
    
    return v_new
