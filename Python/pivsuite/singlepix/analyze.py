"""
Single-pixel PIV analysis module for PIVSuite Python

This module implements the main single-pixel PIV analysis functions.
It corresponds to the pivSinglepixAnalyze.m function in the MATLAB PIVsuite.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from .correlate import singlepix_correlate
from .evaluate import singlepix_evaluate
from .gauss_fit import singlepix_gauss_fit
from .replace import singlepix_replace
from .smooth import singlepix_smooth
from .validate import singlepix_validate
from ..utils.io import load_image


def analyze_singlepix(
    im1: Union[str, np.ndarray],
    im2: Union[str, np.ndarray],
    piv_data: Optional[Dict[str, Any]] = None,
    piv_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform single-pixel PIV analysis on an image pair.
    
    Parameters
    ----------
    im1 : Union[str, np.ndarray]
        First image (either a numpy array or a path to an image file)
    im2 : Union[str, np.ndarray]
        Second image (either a numpy array or a path to an image file)
    piv_data : Optional[Dict[str, Any]]
        Dictionary containing previous PIV results (if any)
    piv_params : Optional[Dict[str, Any]]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing PIV results
    """
    # Initialize parameters if not provided
    if piv_params is None:
        piv_params = {}
    
    # Initialize piv_data if not provided
    if piv_data is None:
        piv_data = {}
    
    # Load images if they are file paths
    if isinstance(im1, str):
        im1_path = im1
        im1 = load_image(im1)
        piv_data['im_filename1'] = im1_path
    
    if isinstance(im2, str):
        im2_path = im2
        im2 = load_image(im2)
        piv_data['im_filename2'] = im2_path
    
    # Get image dimensions
    im_size_y, im_size_x = im1.shape
    piv_data['im_size_x'] = im_size_x
    piv_data['im_size_y'] = im_size_y
    
    # Create grid for single-pixel analysis
    x = np.arange(0, im_size_x)
    y = np.arange(0, im_size_y)
    X, Y = np.meshgrid(x, y)
    
    piv_data['x'] = X
    piv_data['y'] = Y
    
    # Store computation time for each step
    piv_data.setdefault('comp_time', {})
    
    # Step 1: Compute cross-correlation for each pixel
    timer_start = time.time()
    piv_data = singlepix_correlate(im1, im2, piv_data, piv_params)
    piv_data['comp_time']['correlate'] = time.time() - timer_start
    
    # Step 2: Fit Gaussian to cross-correlation peak
    timer_start = time.time()
    piv_data = singlepix_gauss_fit(piv_data, piv_params)
    piv_data['comp_time']['gauss_fit'] = time.time() - timer_start
    
    # Step 3: Evaluate displacement field
    timer_start = time.time()
    piv_data = singlepix_evaluate(piv_data, piv_params)
    piv_data['comp_time']['evaluate'] = time.time() - timer_start
    
    # Step 4: Validate displacement field
    timer_start = time.time()
    piv_data = singlepix_validate(piv_data, piv_params)
    piv_data['comp_time']['validate'] = time.time() - timer_start
    
    # Step 5: Replace invalid vectors
    timer_start = time.time()
    piv_data = singlepix_replace(piv_data, piv_params)
    piv_data['comp_time']['replace'] = time.time() - timer_start
    
    # Step 6: Smooth displacement field
    timer_start = time.time()
    piv_data = singlepix_smooth(piv_data, piv_params)
    piv_data['comp_time']['smooth'] = time.time() - timer_start
    
    return piv_data
