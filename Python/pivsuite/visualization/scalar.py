"""
Scalar plot functions for PIVSuite Python

This module contains functions for creating scalar plots of PIV results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple


def scalar_plot(
    piv_data: Dict[str, Any],
    scalar_data: np.ndarray,
    cmap: str = 'jet',
    title: str = 'Scalar Field',
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    **kwargs
) -> plt.Figure:
    """
    Create a scalar plot of PIV results.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    scalar_data : np.ndarray
        Scalar field to plot
    cmap : str
        Colormap for plot
    title : str
        Title of the plot
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
    # Get coordinates
    x = piv_data['x']
    y = piv_data['y']
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Plot data
    im = plt.pcolormesh(x, y, scalar_data, cmap=cmap, shading='auto', **kwargs)
    
    # Add colorbar
    plt.colorbar(im)
    
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


def contour_plot(
    piv_data: Dict[str, Any],
    scalar_data: np.ndarray,
    levels: Union[int, List[float]] = 10,
    cmap: str = 'jet',
    title: str = 'Contour Plot',
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    **kwargs
) -> plt.Figure:
    """
    Create a contour plot of PIV results.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    scalar_data : np.ndarray
        Scalar field to plot
    levels : Union[int, List[float]]
        Number of contour levels or list of contour levels
    cmap : str
        Colormap for plot
    title : str
        Title of the plot
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    show : bool
        Whether to show the plot
    figsize : Tuple[int, int]
        Figure size in inches
    dpi : int
        DPI for saved image
    **kwargs
        Additional keyword arguments for contour plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    # Get coordinates
    x = piv_data['x']
    y = piv_data['y']
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Plot data
    cs = plt.contour(x, y, scalar_data, levels=levels, cmap=cmap, **kwargs)
    
    # Add colorbar
    plt.colorbar(cs)
    
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
