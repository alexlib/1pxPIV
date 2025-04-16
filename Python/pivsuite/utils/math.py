"""
Math utility functions for PIVSuite Python

This module contains mathematical utility functions for PIV analysis.
"""

import numpy as np
from typing import Tuple, Optional
import numba


@numba.njit
def std_fast(x: np.ndarray) -> float:
    """
    Compute standard deviation of an array.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
        
    Returns
    -------
    float
        Standard deviation
    """
    n = x.size
    mean = np.sum(x) / n
    var = np.sum((x - mean) ** 2) / n
    return np.sqrt(var)


def create_window_function(
    size_x: int,
    size_y: int,
    window_type: str = 'uniform'
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create a window function for cross-correlation.
    
    Parameters
    ----------
    size_x : int
        Size of the window in x direction
    size_y : int
        Size of the window in y direction
    window_type : str
        Type of window function ('uniform', 'parzen', 'hanning', 'gaussian', 'welch')
        
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        W: Window function as a 2D array
        F: Loss-of-correlation function (None for uniform window)
    """
    x = np.linspace(-1, 1, size_x)
    y = np.linspace(-1, 1, size_y)
    X, Y = np.meshgrid(x, y)
    
    if window_type.lower() == 'uniform':
        W = np.ones((size_y, size_x))
        F = None
    
    elif window_type.lower() == 'parzen':
        W = (1 - np.abs(X)) * (1 - np.abs(Y))
        F = np.outer(1 - np.abs(x), 1 - np.abs(y))
    
    elif window_type.lower() == 'hanning':
        W = (0.5 + 0.5 * np.cos(np.pi * X)) * (0.5 + 0.5 * np.cos(np.pi * Y))
        F = np.outer(0.5 + 0.5 * np.cos(np.pi * x), 0.5 + 0.5 * np.cos(np.pi * y))
    
    elif window_type.lower() == 'gaussian':
        sigma = 0.4
        W = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
        F = np.outer(np.exp(-0.5 * x**2 / sigma**2), np.exp(-0.5 * y**2 / sigma**2))
    
    elif window_type.lower() == 'welch':
        W = (1 - X**2) * (1 - Y**2)
        F = np.outer(1 - x**2, 1 - y**2)
    
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    return W, F
