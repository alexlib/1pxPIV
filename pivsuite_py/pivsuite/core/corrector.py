"""
Corrector module for PIVSuite Python

This module handles the predictor-corrector step in PIV analysis.
It corresponds to the pivCorrector.m function in the MATLAB PIVsuite.
"""

import numpy as np
from typing import Dict, Any, Optional


def apply_corrector(
    piv_data: Dict[str, Any],
    piv_data0: Optional[Dict[str, Any]],
    piv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply predictor-corrector to the velocity data.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing current PIV results
    piv_data0 : Optional[Dict[str, Any]]
        Dictionary containing previous PIV results
    piv_params : Dict[str, Any]
        Dictionary containing PIV parameters
        
    Returns
    -------
    Dict[str, Any]
        Updated PIV results
    """
    # If no previous data or no predictor-corrector is needed, return unchanged
    if piv_data0 is None or 'u' not in piv_data0:
        return piv_data
    
    # Get parameters
    pc_method = piv_params.get('pc_method', 'none')
    
    # If no correction is needed, return unchanged
    if pc_method.lower() == 'none':
        return piv_data
    
    # Get velocity fields
    u = piv_data['u']
    v = piv_data['v']
    u0 = piv_data0['u']
    v0 = piv_data0['v']
    
    # Get status arrays
    status = piv_data['status']
    status0 = piv_data0.get('status', np.zeros_like(status))
    
    # Get valid vectors
    valid = (status & 3) == 0  # No mask and no CC failure
    valid0 = (status0 & 3) == 0
    
    # Apply predictor-corrector
    if pc_method.lower() == 'mean':
        # Use mean of current and previous velocity
        u_new = np.copy(u)
        v_new = np.copy(v)
        
        # Only apply to valid vectors
        mask = valid & valid0
        u_new[mask] = 0.5 * (u[mask] + u0[mask])
        v_new[mask] = 0.5 * (v[mask] + v0[mask])
        
        # Update velocity fields
        piv_data['u'] = u_new
        piv_data['v'] = v_new
    
    elif pc_method.lower() == 'weighted':
        # Use weighted mean of current and previous velocity
        u_new = np.copy(u)
        v_new = np.copy(v)
        
        # Get weights
        w = piv_params.get('pc_weight', 0.5)
        
        # Only apply to valid vectors
        mask = valid & valid0
        u_new[mask] = w * u[mask] + (1 - w) * u0[mask]
        v_new[mask] = w * v[mask] + (1 - w) * v0[mask]
        
        # Update velocity fields
        piv_data['u'] = u_new
        piv_data['v'] = v_new
    
    elif pc_method.lower() == 'adaptive':
        # Use adaptive weighting based on correlation peak
        u_new = np.copy(u)
        v_new = np.copy(v)
        
        # Get correlation peaks
        cc_peak = piv_data.get('cc_peak', np.ones_like(u))
        cc_peak0 = piv_data0.get('cc_peak', np.ones_like(u0))
        
        # Only apply to valid vectors
        mask = valid & valid0
        
        # Compute weights
        w = cc_peak / (cc_peak + cc_peak0)
        
        # Apply weights
        u_new[mask] = w[mask] * u[mask] + (1 - w[mask]) * u0[mask]
        v_new[mask] = w[mask] * v[mask] + (1 - w[mask]) * v0[mask]
        
        # Update velocity fields
        piv_data['u'] = u_new
        piv_data['v'] = v_new
    
    return piv_data
