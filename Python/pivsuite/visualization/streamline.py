"""
Streamline plot functions for PIVSuite Python

This module contains functions for creating streamline plots of PIV results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple


def streamline_plot(
    piv_data: Dict[str, Any],
    density: float = 1.0,
    color: str = 'b',
    background_image: Optional[np.ndarray] = None,
    title: str = 'Streamlines',
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    **kwargs
) -> plt.Figure:
    """
    Create a streamline plot of PIV results.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    density : float
        Density of streamlines
    color : str
        Color of streamlines
    background_image : Optional[np.ndarray]
        Background image to display under the streamlines
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
        Additional keyword arguments for streamplot
        
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
    
    # Replace invalid vectors with NaN
    u_valid = u.copy()
    v_valid = v.copy()
    u_valid[~valid] = np.nan
    v_valid[~valid] = np.nan
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Display background image if provided
    if background_image is not None:
        plt.imshow(background_image, cmap='gray', origin='upper')
    
    # Plot streamlines
    plt.streamplot(
        x[0, :], y[:, 0], u_valid, v_valid,
        density=density, color=color, **kwargs
    )
    
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


def colored_streamline_plot(
    piv_data: Dict[str, Any],
    density: float = 1.0,
    cmap: str = 'jet',
    background_image: Optional[np.ndarray] = None,
    title: str = 'Streamlines',
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    **kwargs
) -> plt.Figure:
    """
    Create a colored streamline plot of PIV results.
    
    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    density : float
        Density of streamlines
    cmap : str
        Colormap for streamlines
    background_image : Optional[np.ndarray]
        Background image to display under the streamlines
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
        Additional keyword arguments for streamplot
        
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
    
    # Replace invalid vectors with NaN
    u_valid = u.copy()
    v_valid = v.copy()
    u_valid[~valid] = np.nan
    v_valid[~valid] = np.nan
    
    # Compute velocity magnitude
    magnitude = np.sqrt(u_valid**2 + v_valid**2)
    
    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Display background image if provided
    if background_image is not None:
        plt.imshow(background_image, cmap='gray', origin='upper')
    
    # Plot streamlines
    strm = plt.streamplot(
        x[0, :], y[:, 0], u_valid, v_valid,
        density=density, color=magnitude, cmap=cmap, **kwargs
    )
    
    # Add colorbar
    plt.colorbar(strm.lines, label='Velocity Magnitude')
    
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
