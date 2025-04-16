"""
Parameters module for PIVSuite Python

This module handles the creation and manipulation of PIV parameters.
It corresponds to the pivParams.m function in the MATLAB PIVsuite.
"""

import copy
from typing import Dict, Any, Optional, Union, List


def piv_params(
    piv_data: Optional[Dict[str, Any]] = None,
    piv_params_in: Optional[Dict[str, Any]] = None,
    mode: str = 'defaults',
    pass_no: int = 0
) -> Dict[str, Any]:
    """
    Create or modify PIV parameters.
    
    Parameters
    ----------
    piv_data : Optional[Dict[str, Any]]
        Dictionary containing PIV results
    piv_params_in : Optional[Dict[str, Any]]
        Dictionary containing input PIV parameters
    mode : str
        Mode for parameter creation/modification:
        - 'defaults': Set default values for missing parameters
        - 'singlePass': Extract parameters for a specific pass
    pass_no : int
        Pass number (used only if mode is 'singlePass')
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing PIV parameters
    """
    # Initialize parameters
    if piv_params_in is None:
        piv_params_in = {}
    
    # Create a copy to avoid modifying the input
    piv_par = copy.deepcopy(piv_params_in)
    
    # Set default values for missing parameters
    if mode.lower() == 'defaults':
        # Interrogation area parameters
        piv_par.setdefault('ia_size_x', [32, 16])  # Interrogation area size in x
        piv_par.setdefault('ia_size_y', [32, 16])  # Interrogation area size in y
        piv_par.setdefault('ia_step_x', [16, 8])   # Interrogation area step in x
        piv_par.setdefault('ia_step_y', [16, 8])   # Interrogation area step in y
        piv_par.setdefault('ia_method', 'basic')   # Interrogation method ('basic', 'offset', 'defspline')
        piv_par.setdefault('ia_image_to_deform', 'both')  # Which image to deform ('image1', 'image2', 'both')
        piv_par.setdefault('ia_image_interpolation_method', 'spline')  # Interpolation method ('linear', 'spline')
        piv_par.setdefault('ia_preproc_method', 'none')  # Preprocessing method ('none', 'minmax')
        piv_par.setdefault('ia_min_max_size', 15)  # Size of MinMax filter kernel
        piv_par.setdefault('ia_min_max_level', 0.1)  # Contrast level for MinMax filter
        
        # Cross-correlation parameters
        piv_par.setdefault('cc_remove_ia_mean', 1.0)  # Remove IA mean before cross-correlation
        piv_par.setdefault('cc_max_displacement', 0.7)  # Maximum allowed displacement as fraction of IA size
        piv_par.setdefault('cc_window', 'welch')  # Window function for cross-correlation
        piv_par.setdefault('cc_correct_window_bias', True)  # Correct for bias due to IA windowing
        piv_par.setdefault('cc_method', 'fft')  # Cross-correlation method ('fft', 'dcn')
        piv_par.setdefault('cc_max_dcn_dist', 5)  # Maximum displacement for DCN method
        piv_par.setdefault('cc_subpixel_method', 'gaussian')  # Subpixel interpolation method
        
        # Validation parameters
        piv_par.setdefault('vl_thresh', 2.0)  # Threshold for median test
        piv_par.setdefault('vl_eps', 0.1)  # Epsilon for median test
        piv_par.setdefault('vl_dist', 1)  # Distance for median test
        piv_par.setdefault('vl_passes', 1)  # Number of passes for median test
        
        # Replacement parameters
        piv_par.setdefault('rp_method', 'linear')  # Method for replacing spurious vectors
        
        # Smoothing parameters
        piv_par.setdefault('sm_method', 'none')  # Smoothing method ('none', 'gaussian', 'smoothn')
        piv_par.setdefault('sm_sigma', 1.0)  # Smoothing parameter
        piv_par.setdefault('sm_size', 5)  # Size of smoothing filter
        
        # Analysis parameters
        piv_par.setdefault('an_n_passes', len(piv_par['ia_size_x']))  # Number of passes
        
        # Masking parameters
        piv_par.setdefault('im_mask1', '')  # Mask for first image
        piv_par.setdefault('im_mask2', '')  # Mask for second image
        
        # Visualization parameters
        piv_par.setdefault('qv_pair', {})  # Options for quiver plot
        
    # Extract parameters for a specific pass
    elif mode.lower() == 'singlepass':
        # Make sure pass_no is valid
        if pass_no < 0 or pass_no >= piv_par.get('an_n_passes', 1):
            raise ValueError(f"Invalid pass number: {pass_no}")
        
        # Create a copy of the parameters
        pass_params = copy.deepcopy(piv_par)
        
        # Extract parameters for the specific pass
        for param in ['ia_size_x', 'ia_size_y', 'ia_step_x', 'ia_step_y']:
            if isinstance(piv_par[param], list):
                pass_params[param] = piv_par[param][min(pass_no, len(piv_par[param])-1)]
        
        return pass_params
    
    return piv_par
