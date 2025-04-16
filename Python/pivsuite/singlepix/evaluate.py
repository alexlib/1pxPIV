"""
Single-pixel PIV evaluation module for PIVSuite Python

This module implements the evaluation functions for single-pixel PIV analysis.
It corresponds to the pivSinglepixEvaluate.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple


def singlepix_evaluate(
    piv_data: Dict[str, Any],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate displacement field from single-pixel PIV analysis.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results with evaluated displacement field
    """
    # Get parameters
    min_cc_peak = piv_params.get('sp_min_cc_peak', 0.5)
    
    # Get displacement field
    u = piv_data['u']
    v = piv_data['v']
    cc_peak = piv_data['cc_peak']
    valid_mask = piv_data['valid_mask']
    
    # Create a mask for valid vectors
    valid_vectors = valid_mask & (cc_peak >= min_cc_peak)
    
    # Create a status array (0 = valid, 1 = invalid)
    status = np.zeros_like(u, dtype=np.uint16)
    status[~valid_vectors] = 1
    
    # Store results in piv_data
    piv_data['status'] = status
    piv_data['valid_vectors'] = valid_vectors
    
    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)
    piv_data['magnitude'] = magnitude
    
    # Compute statistics
    piv_data['u_mean'] = np.mean(u[valid_vectors])
    piv_data['v_mean'] = np.mean(v[valid_vectors])
    piv_data['u_std'] = np.std(u[valid_vectors])
    piv_data['v_std'] = np.std(v[valid_vectors])
    piv_data['magnitude_mean'] = np.mean(magnitude[valid_vectors])
    piv_data['magnitude_std'] = np.std(magnitude[valid_vectors])
    
    return piv_data
