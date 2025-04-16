"""
Replacement module for PIVSuite Python

This module handles the replacement of invalid vectors.
It corresponds to the pivReplace.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.interpolate import griddata


def replace_vectors(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Replace invalid vectors in the velocity field.

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
    # Get velocity fields
    u = piv_data['u']
    v = piv_data['v']

    # Get coordinates
    x = piv_data['x']
    y = piv_data['y']

    # Get status array
    status = piv_data['status']

    # Get replacement parameters
    rp_method = piv_params.get('rp_method', 'linear')

    # If no replacement is needed, return unchanged
    if rp_method.lower() == 'none':
        return piv_data

    # Get valid vectors (not masked, no CC failure, no validation failure)
    valid = (status & 11) == 0  # 11 = 1 + 2 + 8

    # Create a copy of the velocity fields
    u_replaced = np.copy(u)
    v_replaced = np.copy(v)

    # Replace invalid vectors
    if rp_method.lower() == 'linear':
        # Use linear interpolation
        u_replaced, v_replaced = linear_interpolation(x, y, u, v, valid)

    elif rp_method.lower() == 'nearest':
        # Use nearest neighbor interpolation
        u_replaced, v_replaced = nearest_interpolation(x, y, u, v, valid)

    elif rp_method.lower() == 'spline':
        # Use spline interpolation
        u_replaced, v_replaced = spline_interpolation(x, y, u, v, valid)

    elif rp_method.lower() == 'mean':
        # Use mean of valid vectors
        u_mean = np.nanmean(u[valid])
        v_mean = np.nanmean(v[valid])

        u_replaced[~valid] = u_mean
        v_replaced[~valid] = v_mean

    # Store results in piv_data
    piv_data['u_original'] = u
    piv_data['v_original'] = v
    piv_data['u'] = u_replaced
    piv_data['v'] = v_replaced

    # Print debug information
    print(f"Valid vectors before replacement: {np.sum(valid)} out of {valid.size}")
    print(f"Spurious vectors before replacement: {piv_data['spurious_n']}")

    return piv_data


def linear_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace invalid vectors using linear interpolation.

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
        Mask of valid vectors

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_replaced: x-component of velocity field with replaced vectors
        v_replaced: y-component of velocity field with replaced vectors
    """
    # Create a copy of the velocity fields
    u_replaced = np.copy(u)
    v_replaced = np.copy(v)

    # Check if there are any invalid vectors
    if np.all(valid):
        return u_replaced, v_replaced

    # Check if there are enough valid vectors for interpolation
    if np.sum(valid) < 3:
        # Not enough valid vectors, use mean
        u_mean = np.nanmean(u[valid])
        v_mean = np.nanmean(v[valid])

        u_replaced[~valid] = u_mean
        v_replaced[~valid] = v_mean

        return u_replaced, v_replaced

    # Get coordinates of valid vectors
    x_valid = x[valid]
    y_valid = y[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Get coordinates of invalid vectors
    x_invalid = x[~valid]
    y_invalid = y[~valid]

    # Interpolate u and v at invalid positions
    u_interp = griddata((x_valid, y_valid), u_valid, (x_invalid, y_invalid), method='linear')
    v_interp = griddata((x_valid, y_valid), v_valid, (x_invalid, y_invalid), method='linear')

    # Replace invalid vectors with interpolated values
    u_replaced[~valid] = u_interp
    v_replaced[~valid] = v_interp

    # Handle any remaining NaN values
    u_replaced = np.nan_to_num(u_replaced, nan=np.nanmean(u_valid))
    v_replaced = np.nan_to_num(v_replaced, nan=np.nanmean(v_valid))

    return u_replaced, v_replaced


def nearest_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace invalid vectors using nearest neighbor interpolation.

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
        Mask of valid vectors

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_replaced: x-component of velocity field with replaced vectors
        v_replaced: y-component of velocity field with replaced vectors
    """
    # Create a copy of the velocity fields
    u_replaced = np.copy(u)
    v_replaced = np.copy(v)

    # Check if there are any invalid vectors
    if np.all(valid):
        return u_replaced, v_replaced

    # Check if there are enough valid vectors for interpolation
    if np.sum(valid) < 1:
        # Not enough valid vectors, use mean
        u_mean = np.nanmean(u)
        v_mean = np.nanmean(v)

        u_replaced[~valid] = u_mean
        v_replaced[~valid] = v_mean

        return u_replaced, v_replaced

    # Get coordinates of valid vectors
    x_valid = x[valid]
    y_valid = y[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Get coordinates of invalid vectors
    x_invalid = x[~valid]
    y_invalid = y[~valid]

    # Interpolate u and v at invalid positions
    u_interp = griddata((x_valid, y_valid), u_valid, (x_invalid, y_invalid), method='nearest')
    v_interp = griddata((x_valid, y_valid), v_valid, (x_invalid, y_invalid), method='nearest')

    # Replace invalid vectors with interpolated values
    u_replaced[~valid] = u_interp
    v_replaced[~valid] = v_interp

    return u_replaced, v_replaced


def spline_interpolation(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replace invalid vectors using spline interpolation.

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
        Mask of valid vectors

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u_replaced: x-component of velocity field with replaced vectors
        v_replaced: y-component of velocity field with replaced vectors
    """
    # Create a copy of the velocity fields
    u_replaced = np.copy(u)
    v_replaced = np.copy(v)

    # Check if there are any invalid vectors
    if np.all(valid):
        return u_replaced, v_replaced

    # Check if there are enough valid vectors for interpolation
    if np.sum(valid) < 4:
        # Not enough valid vectors, use linear interpolation
        return linear_interpolation(x, y, u, v, valid)

    # Get coordinates of valid vectors
    x_valid = x[valid]
    y_valid = y[valid]
    u_valid = u[valid]
    v_valid = v[valid]

    # Get coordinates of invalid vectors
    x_invalid = x[~valid]
    y_invalid = y[~valid]

    # Interpolate u and v at invalid positions
    u_interp = griddata((x_valid, y_valid), u_valid, (x_invalid, y_invalid), method='cubic')
    v_interp = griddata((x_valid, y_valid), v_valid, (x_invalid, y_invalid), method='cubic')

    # Replace invalid vectors with interpolated values
    u_replaced[~valid] = u_interp
    v_replaced[~valid] = v_interp

    # Handle any remaining NaN values
    u_replaced = np.nan_to_num(u_replaced, nan=np.nanmean(u_valid))
    v_replaced = np.nan_to_num(v_replaced, nan=np.nanmean(v_valid))

    return u_replaced, v_replaced
