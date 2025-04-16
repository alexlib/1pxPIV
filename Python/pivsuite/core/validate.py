"""
Validation module for PIVSuite Python

This module handles the validation of velocity vectors.
It corresponds to the pivValidate.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy import ndimage


def validate_velocity(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate velocity field and mark spurious vectors.

    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters

    Returns
    -------
    Dict[str, Any]
        Updated PIV results with validated vectors
    """
    # Get velocity fields
    u = piv_data['u']
    v = piv_data['v']

    # Get status array
    status = piv_data['status']

    # Get validation parameters
    vl_thresh = piv_params.get('vl_thresh', 2.0)
    vl_eps = piv_params.get('vl_eps', 0.1)
    vl_dist = piv_params.get('vl_dist', 1)
    vl_passes = piv_params.get('vl_passes', 1)

    # Create a copy of the status array
    status_new = status.copy()

    # Get valid vectors (not masked and no CC failure)
    valid = (status & 3) == 0
    print(f"Initial valid vectors: {np.sum(valid)} out of {status.size}")

    # Apply median test
    for _ in range(vl_passes):
        # Create a copy of the valid array
        valid_new = valid.copy()

        # Apply median test
        valid_new = median_test(u, v, valid, vl_thresh, vl_eps, vl_dist)

        # Update status array
        status_new[~valid_new & valid] |= 8  # Set bit 3 (validation failed)

        # Update valid array
        valid = valid_new

    # Count spurious vectors
    spurious_n = np.sum(~valid & ((status & 3) == 0))
    print(f"Spurious vectors after validation: {spurious_n}")

    # Store results in piv_data
    piv_data['status'] = status_new
    piv_data['spurious_n'] = spurious_n

    return piv_data


def median_test(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    threshold: float,
    epsilon: float,
    distance: int
) -> np.ndarray:
    """
    Apply median test to velocity field.

    Parameters
    ----------
    u : np.ndarray
        x-component of velocity field
    v : np.ndarray
        y-component of velocity field
    valid : np.ndarray
        Mask of valid vectors
    threshold : float
        Threshold for median test
    epsilon : float
        Epsilon for median test
    distance : int
        Distance for median test

    Returns
    -------
    np.ndarray
        Updated mask of valid vectors
    """
    # Create a copy of the valid array
    valid_new = valid.copy()
    print(f"Before median test: {np.sum(valid_new)} valid vectors")

    # Get the shape of the velocity field
    ny, nx = u.shape

    # Create a kernel for the median filter
    kernel_size = 2 * distance + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    kernel[distance, distance] = False  # Exclude the center point

    # Loop through all vectors
    for i in range(ny):
        for j in range(nx):
            # Skip invalid vectors
            if not valid[i, j]:
                continue

            # Define the neighborhood
            i_min = max(0, i - distance)
            i_max = min(ny, i + distance + 1)
            j_min = max(0, j - distance)
            j_max = min(nx, j + distance + 1)

            # Get the neighborhood vectors
            u_neighbors = u[i_min:i_max, j_min:j_max]
            v_neighbors = v[i_min:i_max, j_min:j_max]
            valid_neighbors = valid[i_min:i_max, j_min:j_max]

            # Skip if there are not enough valid neighbors
            if np.sum(valid_neighbors) <= 1:
                continue

            # Exclude the center point
            center_i = i - i_min
            center_j = j - j_min
            mask = np.ones_like(valid_neighbors, dtype=bool)
            mask[center_i, center_j] = False

            # Get valid neighbors
            valid_mask = valid_neighbors & mask

            # Skip if there are not enough valid neighbors
            if np.sum(valid_mask) < 1:
                continue

            # Calculate median and residuals
            u_median = np.median(u_neighbors[valid_mask])
            v_median = np.median(v_neighbors[valid_mask])

            # Calculate residuals
            u_res = np.abs(u[i, j] - u_median)
            v_res = np.abs(v[i, j] - v_median)

            # Calculate median of residuals
            u_res_median = np.median(np.abs(u_neighbors[valid_mask] - u_median))
            v_res_median = np.median(np.abs(v_neighbors[valid_mask] - v_median))

            # Apply threshold
            u_threshold = max(threshold * u_res_median, epsilon)
            v_threshold = max(threshold * v_res_median, epsilon)

            # Mark as invalid if residuals exceed threshold
            if u_res > u_threshold or v_res > v_threshold:
                valid_new[i, j] = False

    print(f"After median test: {np.sum(valid_new)} valid vectors")
    return valid_new
