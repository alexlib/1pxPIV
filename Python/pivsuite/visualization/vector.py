"""
Vector plot functions for PIVSuite Python

This module contains functions for creating vector plots of PIV results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple


def vector_plot(
    piv_data: Dict[str, Any],
    component: str = 'magnitude',
    cmap: str = 'jet',
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    **kwargs
) -> plt.Figure:
    """
    Create a vector plot of PIV results.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    component : str
        Component to plot ('u', 'v', 'magnitude', 'vorticity', 'divergence')
    cmap : str
        Colormap for plot
    title : Optional[str]
        Title of the plot (if None, use component name)
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    show : bool
        Whether to show the plot
    figsize : Tuple[int, int]
        Figure size in inches
    dpi : int
        DPI for saved image
    **kwargs
        Additional keyword arguments for pcolormesh plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Get velocity fields
    x = piv_data['x']
    y = piv_data['y']
    u = piv_data['u']
    v = piv_data['v']
    
    # Get status array
    status = piv_data.get('status', np.zeros_like(u, dtype=np.uint16))
    
    # Create mask for valid vectors
    valid = (status & 11) == 0  # 11 = 1 + 2 + 8
    
    # Compute the requested component
    if component.lower() == 'u':
        data = u
        if title is None:
            title = 'X Velocity Component'
    elif component.lower() == 'v':
        data = v
        if title is None:
            title = 'Y Velocity Component'
    elif component.lower() == 'magnitude':
        data = np.sqrt(u**2 + v**2)
        if title is None:
            title = 'Velocity Magnitude'
    elif component.lower() == 'vorticity':
        # Compute grid spacing
        dx = np.mean(np.diff(x[0, :]))
        dy = np.mean(np.diff(y[:, 0]))
        
        # Compute derivatives
        du_dy, du_dx = np.gradient(u, dy, dx)
        dv_dy, dv_dx = np.gradient(v, dy, dx)
        
        # Compute vorticity
        data = dv_dx - du_dy
        if title is None:
            title = 'Vorticity'
    elif component.lower() == 'divergence':
        # Compute grid spacing
        dx = np.mean(np.diff(x[0, :]))
        dy = np.mean(np.diff(y[:, 0]))
        
        # Compute derivatives
        du_dy, du_dx = np.gradient(u, dy, dx)
        dv_dy, dv_dx = np.gradient(v, dy, dx)
        
        # Compute divergence
        data = du_dx + dv_dy
        if title is None:
            title = 'Divergence'
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Plot data
    im = plt.pcolormesh(x, y, data, cmap=cmap, shading='auto', **kwargs)
    
    # Add colorbar
    plt.colorbar(im, label=component.capitalize())
    
    # Add title and adjust plot
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
    
    # Show if requested
    if show:
        plt.show()
    
    return fig
