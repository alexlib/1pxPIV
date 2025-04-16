"""
Analysis module for PIVSuite Python

This module implements the main PIV analysis functions.
It corresponds to the pivAnalyzeImagePair.m function in the MATLAB PIVsuite.
"""

import numpy as np
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from .params import piv_params
from .interrogate import interrogate_images
from .cross_corr import cross_correlate
from .corrector import apply_corrector
from .validate import validate_velocity
from .replace import replace_vectors
from .smooth import smooth_velocity
from ..utils.io import load_image


def analyze_image_pair(
    im1: Union[str, np.ndarray],
    im2: Union[str, np.ndarray],
    piv_data: Optional[Dict[str, Any]] = None,
    piv_params_in: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Perform single- or multi-pass analysis of displacement between two images using PIV technique.
    
    Parameters
    ----------
    im1 : Union[str, np.ndarray]
        First image (either a numpy array or a path to an image file)
    im2 : Union[str, np.ndarray]
        Second image (either a numpy array or a path to an image file)
    piv_data : Optional[Dict[str, Any]]
        Dictionary containing previous PIV results (if any)
    piv_params_in : Optional[Dict[str, Any]]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Tuple[Dict[str, Any], Optional[np.ndarray]]
        piv_data: Dictionary containing PIV results
        cc_peak_im: Cross-correlation peak image (optional)
    """
    # Initialize parameters if not provided
    if piv_params_in is None:
        piv_params_in = {}
    
    # Set default parameters
    piv_par = piv_params(None, piv_params_in, 'defaults')
    
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
    
    # Store computation time for each pass
    piv_data.setdefault('comp_time', [])
    
    # Loop for all required passes
    cc_peak_im = None
    for pass_idx in range(piv_par['an_n_passes']):
        timer_start = time.time()
        
        # Save velocity before computation - will be used if predictor-corrector is used
        piv_data0 = piv_data.copy() if 'u' in piv_data else None
        
        # Extract parameters for the current pass
        pass_params = piv_params(piv_data, piv_par, 'singlePass', pass_idx)
        
        # Find interrogation areas in images, shift or deform them if required
        ex_im1, ex_im2, piv_data = interrogate_images(im1, im2, piv_data, pass_params)
        
        # Compute cross-correlation between interrogation areas
        if pass_idx == piv_par['an_n_passes'] - 1 and piv_par.get('return_cc_peak_im', False):
            piv_data, cc_peak_im = cross_correlate(ex_im1, ex_im2, piv_data, pass_params)
        else:
            piv_data = cross_correlate(ex_im1, ex_im2, piv_data, pass_params)
        
        # Apply predictor-corrector to the velocity data
        piv_data = apply_corrector(piv_data, piv_data0, pass_params)
        
        # Validate velocity field
        piv_data = validate_velocity(piv_data, pass_params)
        
        # Interpolate invalid velocity vectors
        piv_data = replace_vectors(piv_data, pass_params)
        
        # Smooth the velocity field
        piv_data = smooth_velocity(piv_data, pass_params)
        
        # Save the information about the current pass
        piv_data['pass_no'] = pass_idx + 1
        
        # Record computation time
        piv_data['comp_time'].append(time.time() - timer_start)
        
        # Print progress
        print(f"Pass {pass_idx+1}/{piv_par['an_n_passes']} completed in {piv_data['comp_time'][-1]:.2f} seconds")
    
    return piv_data, cc_peak_im


def analyze_image_sequence(
    im1_list: List[Union[str, np.ndarray]],
    im2_list: List[Union[str, np.ndarray]],
    piv_data: Optional[Dict[str, Any]] = None,
    piv_params_in: Optional[Dict[str, Any]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze a sequence of image pairs using PIV.
    
    Parameters
    ----------
    im1_list : List[Union[str, np.ndarray]]
        List of first images in each pair
    im2_list : List[Union[str, np.ndarray]]
        List of second images in each pair
    piv_data : Optional[Dict[str, Any]]
        Dictionary containing previous PIV results (if any)
    piv_params_in : Optional[Dict[str, Any]]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary containing PIV results for each image pair
    """
    if len(im1_list) != len(im2_list):
        raise ValueError("Number of first and second images must be the same")
    
    # Initialize parameters if not provided
    if piv_params_in is None:
        piv_params_in = {}
    
    # Set default parameters
    piv_par = piv_params(None, piv_params_in, 'defaults')
    
    # Initialize results
    piv_data_seq = {'results': []}
    
    # Process each image pair
    for i, (im1, im2) in enumerate(zip(im1_list, im2_list)):
        print(f"\nProcessing image pair {i+1}/{len(im1_list)}")
        
        # Initialize data for this pair
        pair_data = piv_data.copy() if piv_data is not None else {}
        
        # Analyze image pair
        pair_result, _ = analyze_image_pair(im1, im2, pair_data, piv_par)
        
        # Store results
        piv_data_seq['results'].append(pair_result)
        
        # Use current result as initial guess for next pair if requested
        if piv_par.get('use_previous_as_estimate', False) and i < len(im1_list) - 1:
            piv_data = pair_result
    
    return piv_data_seq
