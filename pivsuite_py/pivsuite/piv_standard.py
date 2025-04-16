"""
Standard PIV algorithm implementation

This module implements the standard PIV algorithm for analyzing image pairs.
It corresponds to the pivAnalyzeImagePair.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Tuple, List, Dict, Union, Optional, Any
import time
from tqdm import tqdm

from .utils import (
    load_image, 
    create_window_function, 
    cross_correlate_fft, 
    cross_correlate_direct,
    find_peak_position,
    validate_vectors,
    replace_invalid_vectors,
    smooth_vector_field,
    compute_vorticity
)

from .interrogation import interrogate_images
from .cross_correlation import compute_cross_correlation


def analyze_image_pair(
    im1: Union[str, np.ndarray], 
    im2: Union[str, np.ndarray], 
    piv_data: Optional[Dict[str, Any]] = None, 
    piv_params: Optional[Dict[str, Any]] = None
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
    piv_params : Optional[Dict[str, Any]]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Tuple[Dict[str, Any], Optional[np.ndarray]]
        piv_data: Dictionary containing PIV results
        cc_function: Cross-correlation function (optional)
    """
    # Initialize parameters if not provided
    if piv_params is None:
        piv_params = {}
    
    # Set default parameters if not specified
    piv_params.setdefault('ia_size_x', [32, 16])  # Interrogation area size in x
    piv_params.setdefault('ia_size_y', [32, 16])  # Interrogation area size in y
    piv_params.setdefault('ia_step_x', [16, 8])   # Interrogation area step in x
    piv_params.setdefault('ia_step_y', [16, 8])   # Interrogation area step in y
    piv_params.setdefault('ia_method', 'basic')   # Interrogation method ('basic', 'offset', 'deform')
    piv_params.setdefault('cc_method', 'fft')     # Cross-correlation method ('fft', 'direct')
    piv_params.setdefault('cc_window', 'uniform') # Window function for cross-correlation
    piv_params.setdefault('cc_max_displacement', 0.7) # Max displacement as fraction of IA size
    piv_params.setdefault('cc_subpixel_method', 'gaussian') # Subpixel interpolation method
    piv_params.setdefault('vl_method', 'median')  # Validation method
    piv_params.setdefault('vl_threshold', 2.0)    # Validation threshold
    piv_params.setdefault('rp_method', 'interpolate') # Replacement method for invalid vectors
    piv_params.setdefault('sm_method', 'gaussian') # Smoothing method
    piv_params.setdefault('sm_sigma', 1.0)        # Smoothing parameter
    piv_params.setdefault('n_passes', len(piv_params['ia_size_x'])) # Number of passes
    
    # Initialize piv_data if not provided
    if piv_data is None:
        piv_data = {}
    
    # Load images if they are file paths
    if isinstance(im1, str):
        im1 = load_image(im1)
        piv_data['im_filename1'] = im1
    
    if isinstance(im2, str):
        im2 = load_image(im2)
        piv_data['im_filename2'] = im2
    
    # Store computation time for each pass
    piv_data.setdefault('comp_time', [])
    
    # Loop for all required passes
    cc_function = None
    for pass_idx in range(piv_params['n_passes']):
        timer_start = time.time()
        
        # Save velocity before computation - will be used if predictor-corrector is used
        piv_data0 = piv_data.copy() if 'u' in piv_data else None
        
        # Extract parameters for the current pass
        pass_params = extract_pass_params(piv_params, pass_idx)
        
        # Find interrogation areas in images, shift or deform them if required
        ex_im1, ex_im2, piv_data = interrogate_images(im1, im2, piv_data, pass_params)
        
        # Compute cross-correlation between interrogation areas
        if pass_idx == piv_params['n_passes'] - 1:
            piv_data, cc_function = compute_cross_correlation(ex_im1, ex_im2, piv_data, pass_params)
        else:
            piv_data = compute_cross_correlation(ex_im1, ex_im2, piv_data, pass_params)
        
        # Apply predictor-corrector to the velocity data
        piv_data = apply_corrector(piv_data, piv_data0, pass_params)
        
        # Validate velocity field
        piv_data = validate_velocity_field(piv_data, pass_params)
        
        # Interpolate invalid velocity vectors
        piv_data = replace_invalid_vectors_in_data(piv_data, pass_params)
        
        # Smooth the velocity field
        piv_data = smooth_velocity_field(piv_data, pass_params)
        
        # Save the information about the current pass
        piv_data['pass_no'] = pass_idx + 1
        
        # Record computation time
        piv_data['comp_time'].append(time.time() - timer_start)
        
        # Print progress
        print(f"Pass {pass_idx+1}/{piv_params['n_passes']} completed in {piv_data['comp_time'][-1]:.2f} seconds")
    
    return piv_data, cc_function


def extract_pass_params(piv_params: Dict[str, Any], pass_idx: int) -> Dict[str, Any]:
    """
    Extract parameters for a specific pass.
    
    Parameters
    ----------
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
    pass_idx : int
        Index of the current pass
        
    Returns
    -------
    Dict[str, Any]
        Parameters for the current pass
    """
    pass_params = piv_params.copy()
    
    # Extract parameters that can vary between passes
    for param in ['ia_size_x', 'ia_size_y', 'ia_step_x', 'ia_step_y']:
        if isinstance(piv_params[param], list):
            pass_params[param] = piv_params[param][min(pass_idx, len(piv_params[param])-1)]
    
    return pass_params


def apply_corrector(piv_data: Dict[str, Any], piv_data0: Optional[Dict[str, Any]], 
                   pass_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply predictor-corrector to the velocity data.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing current PIV results
    piv_data0 : Optional[Dict[str, Any]]
        Dictionary containing previous PIV results
    pass_params : Dict[str, Any]
        Parameters for the current pass
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results
    """
    # If no previous data or no predictor-corrector is needed, return unchanged
    if piv_data0 is None or 'u' not in piv_data0:
        return piv_data
    
    # If predictor-corrector method is specified, apply it
    if 'pc_method' in pass_params and pass_params['pc_method'] != 'none':
        # Implement predictor-corrector methods here
        # For now, we'll just return the data unchanged
        pass
    
    return piv_data


def validate_velocity_field(piv_data: Dict[str, Any], pass_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate velocity field and mark spurious vectors.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    pass_params : Dict[str, Any]
        Parameters for the current pass
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with validated vectors
    """
    if 'u' not in piv_data or 'v' not in piv_data:
        return piv_data
    
    # Get validation parameters
    method = pass_params.get('vl_method', 'median')
    threshold = pass_params.get('vl_threshold', 2.0)
    
    # Validate vectors
    valid = validate_vectors(piv_data['u'], piv_data['v'], method, threshold)
    
    # Store validation results
    piv_data['valid'] = valid
    piv_data['spurious_n'] = np.sum(~valid)
    
    return piv_data


def replace_invalid_vectors_in_data(piv_data: Dict[str, Any], pass_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace invalid vectors in the velocity field.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    pass_params : Dict[str, Any]
        Parameters for the current pass
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with replaced vectors
    """
    if 'u' not in piv_data or 'v' not in piv_data or 'valid' not in piv_data:
        return piv_data
    
    # Get replacement parameters
    method = pass_params.get('rp_method', 'interpolate')
    
    # Replace invalid vectors
    u_replaced, v_replaced = replace_invalid_vectors(
        piv_data['x'], piv_data['y'], 
        piv_data['u'], piv_data['v'], 
        piv_data['valid'], method
    )
    
    # Store replaced vectors
    piv_data['u_replaced'] = u_replaced
    piv_data['v_replaced'] = v_replaced
    
    # Update u and v with replaced values
    piv_data['u'] = u_replaced
    piv_data['v'] = v_replaced
    
    return piv_data


def smooth_velocity_field(piv_data: Dict[str, Any], pass_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Smooth the velocity field.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    pass_params : Dict[str, Any]
        Parameters for the current pass
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with smoothed vectors
    """
    if 'u' not in piv_data or 'v' not in piv_data:
        return piv_data
    
    # Get smoothing parameters
    method = pass_params.get('sm_method', 'gaussian')
    sigma = pass_params.get('sm_sigma', 1.0)
    
    # Skip smoothing if method is 'none'
    if method.lower() == 'none':
        return piv_data
    
    # Smooth velocity field
    u_smoothed, v_smoothed = smooth_vector_field(
        piv_data['u'], piv_data['v'], method, sigma
    )
    
    # Store original vectors before smoothing
    piv_data['u_original'] = piv_data['u'].copy()
    piv_data['v_original'] = piv_data['v'].copy()
    
    # Update u and v with smoothed values
    piv_data['u'] = u_smoothed
    piv_data['v'] = v_smoothed
    
    return piv_data


def analyze_image_sequence(
    im1_list: List[Union[str, np.ndarray]], 
    im2_list: List[Union[str, np.ndarray]], 
    piv_data: Optional[Dict[str, Any]] = None, 
    piv_params: Optional[Dict[str, Any]] = None
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
    piv_params : Optional[Dict[str, Any]]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary containing PIV results for each image pair
    """
    if len(im1_list) != len(im2_list):
        raise ValueError("Number of first and second images must be the same")
    
    # Initialize parameters if not provided
    if piv_params is None:
        piv_params = {}
    
    # Initialize results
    piv_data_seq = {'results': []}
    
    # Process each image pair
    for i, (im1, im2) in enumerate(tqdm(zip(im1_list, im2_list), total=len(im1_list))):
        print(f"\nProcessing image pair {i+1}/{len(im1_list)}")
        
        # Initialize data for this pair
        pair_data = piv_data.copy() if piv_data is not None else {}
        
        # Analyze image pair
        pair_result, _ = analyze_image_pair(im1, im2, pair_data, piv_params)
        
        # Store results
        piv_data_seq['results'].append(pair_result)
        
        # Use current result as initial guess for next pair if requested
        if piv_params.get('use_previous_as_estimate', False) and i < len(im1_list) - 1:
            piv_data = pair_result
    
    return piv_data_seq
