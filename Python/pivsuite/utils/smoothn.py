"""
Smoothn algorithm for PIVSuite Python

This module implements the smoothn algorithm for smoothing noisy data.
It is a Python port of the MATLAB smoothn function by Damien Garcia.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional, Union, List


def smoothn(
    y: np.ndarray,
    s: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    robust: bool = False,
    max_iter: int = 100,
    tol: float = 1e-3
) -> Tuple[np.ndarray, float]:
    """
    Smooth noisy data using the smoothn algorithm.
    
    Parameters
    ----------
    y : np.ndarray
        Data to be smoothed
    s : Optional[float]
        Smoothing parameter (if None, it is automatically determined)
    weights : Optional[np.ndarray]
        Weights for each data point (if None, all weights are 1)
    robust : bool
        Whether to use robust smoothing (for data with outliers)
    max_iter : int
        Maximum number of iterations for robust smoothing
    tol : float
        Tolerance for convergence
        
    Returns
    -------
    Tuple[np.ndarray, float]
        z: Smoothed data
        s: Smoothing parameter used
    """
    # Get data shape and dimensions
    shape = y.shape
    ndims = len(shape)
    
    # Reshape data to 2D array
    y_flat = y.reshape(-1, 1)
    n_elements = y_flat.shape[0]
    
    # Initialize weights
    if weights is None:
        weights = np.ones_like(y_flat)
    else:
        weights = weights.reshape(-1, 1)
    
    # Initialize smoothing parameter
    if s is None:
        s = 1.0
    
    # Create differential operators
    d_operators = []
    for i in range(ndims):
        shape_i = np.ones(ndims, dtype=int)
        shape_i[i] = shape[i]
        d_operators.append(diff_operator(shape[i]).reshape(shape_i))
    
    # Create regularization matrix
    Lambda = np.zeros(shape)
    for i in range(ndims):
        Lambda = Lambda + np.abs(np.fft.fftn(d_operators[i])) ** 2
    
    # Smooth data
    z = y_flat.copy()
    
    if robust:
        # Robust smoothing
        iter_count = 0
        prev_weights = np.zeros_like(weights)
        
        while iter_count < max_iter and np.max(np.abs(weights - prev_weights)) > tol:
            prev_weights = weights.copy()
            
            # Smooth with current weights
            z = smooth_weighted(y_flat, weights, Lambda, s)
            
            # Compute residuals
            res = y_flat - z
            
            # Compute robust weights
            mad = np.median(np.abs(res - np.median(res)))
            weights = bisquare(res / (1.4826 * mad))
            
            iter_count += 1
    else:
        # Non-robust smoothing
        z = smooth_weighted(y_flat, weights, Lambda, s)
    
    # Reshape smoothed data to original shape
    z = z.reshape(shape)
    
    return z, s


def smooth_weighted(
    y: np.ndarray,
    weights: np.ndarray,
    Lambda: np.ndarray,
    s: float
) -> np.ndarray:
    """
    Smooth data with weights.
    
    Parameters
    ----------
    y : np.ndarray
        Data to be smoothed
    weights : np.ndarray
        Weights for each data point
    Lambda : np.ndarray
        Regularization matrix
    s : float
        Smoothing parameter
        
    Returns
    -------
    np.ndarray
        Smoothed data
    """
    # Compute weighted data
    y_weighted = y * weights
    
    # Compute DCT of weighted data
    y_dct = np.fft.fftn(y_weighted.reshape(Lambda.shape))
    
    # Compute smoothed data
    z_dct = y_dct / (1 + s * Lambda)
    z = np.real(np.fft.ifftn(z_dct)).reshape(y.shape)
    
    return z


def diff_operator(n: int) -> np.ndarray:
    """
    Create a 1D differential operator.
    
    Parameters
    ----------
    n : int
        Size of the operator
        
    Returns
    -------
    np.ndarray
        Differential operator
    """
    # Create differential operator
    d = np.zeros(n)
    d[0] = 1
    d[1] = -2
    d[2] = 1
    
    return d


def bisquare(x: np.ndarray) -> np.ndarray:
    """
    Bisquare (Tukey) weight function.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
        
    Returns
    -------
    np.ndarray
        Weights
    """
    # Compute weights
    weights = np.zeros_like(x)
    idx = np.abs(x) < 1
    weights[idx] = (1 - x[idx]**2) ** 2
    
    return weights
