"""
Visualization module for PIVSuite

This module provides functions for visualizing PIV and BOS results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Tuple, Dict, Union, Optional, Any


def quiver_plot(
    results: Dict[str, Any],
    background_image: Optional[np.ndarray] = None,
    scale: float = 1.0,
    width: float = 0.005,
    color: str = 'r',
    autoscale: bool = False,
    title: str = 'Velocity Field',
    output_path: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create a quiver plot of velocity vectors.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing PIV or BOS results
    background_image : Optional[np.ndarray]
        Background image to display under the vectors
    scale : float
        Scale factor for arrows
    width : float
        Width of arrows
    color : str
        Color of arrows
    autoscale : bool
        Whether to autoscale the arrows
    title : str
        Title of the plot
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    dpi : int
        DPI for saved image
    figsize : Tuple[int, int]
        Figure size in inches
        
    Returns
    -------
    None
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display background image if provided
    if background_image is not None:
        plt.imshow(background_image, cmap='gray')
        plt.colormap('gray')
    
    # Plot quiver
    if autoscale:
        plt.quiver(x, y, u, v, color=color, width=width)
    else:
        plt.quiver(x, y, u, v, color=color, width=width, scale=1.0/scale)
    
    # Add title and adjust plot
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to {output_path}")
    
    plt.show()


def vector_plot(
    results: Dict[str, Any],
    component: str = 'magnitude',
    colormap: str = 'jet',
    show_colorbar: bool = True,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> None:
    """
    Create a color plot of velocity field components.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing PIV or BOS results
    component : str
        Component to plot ('u', 'v', 'magnitude', 'divergence', 'curl')
    colormap : str
        Colormap to use
    show_colorbar : bool
        Whether to show the colorbar
    title : Optional[str]
        Title of the plot (if None, use component name)
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    dpi : int
        DPI for saved image
    figsize : Tuple[int, int]
        Figure size in inches
    vmin : Optional[float]
        Minimum value for color scale
    vmax : Optional[float]
        Maximum value for color scale
        
    Returns
    -------
    None
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    
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
    elif component.lower() == 'divergence':
        # Compute grid spacing
        dx = np.mean(np.diff(x[0, :]))
        dy = np.mean(np.diff(y[:, 0]))
        
        # Compute derivatives
        du_dx = np.gradient(u, dx, axis=1)
        dv_dy = np.gradient(v, dy, axis=0)
        
        # Compute divergence
        data = du_dx + dv_dy
        if title is None:
            title = 'Velocity Divergence'
    elif component.lower() == 'curl' or component.lower() == 'vorticity':
        # Compute grid spacing
        dx = np.mean(np.diff(x[0, :]))
        dy = np.mean(np.diff(y[:, 0]))
        
        # Compute derivatives
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)
        
        # Compute curl
        data = dv_dx - du_dy
        if title is None:
            title = 'Vorticity'
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create color plot
    im = plt.pcolormesh(x, y, data, cmap=colormap, shading='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im)
    
    # Add title and adjust plot
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to {output_path}")
    
    plt.show()


def vorticity_plot(
    results: Dict[str, Any],
    colormap: str = 'RdBu_r',
    show_colorbar: bool = True,
    title: str = 'Vorticity',
    output_path: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_vectors: bool = False,
    vector_color: str = 'k',
    vector_scale: float = 1.0,
    vector_width: float = 0.005,
    vector_skip: int = 2
) -> None:
    """
    Create a vorticity plot with optional velocity vectors.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing PIV or BOS results
    colormap : str
        Colormap to use
    show_colorbar : bool
        Whether to show the colorbar
    title : str
        Title of the plot
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    dpi : int
        DPI for saved image
    figsize : Tuple[int, int]
        Figure size in inches
    vmin : Optional[float]
        Minimum value for color scale
    vmax : Optional[float]
        Maximum value for color scale
    show_vectors : bool
        Whether to overlay velocity vectors
    vector_color : str
        Color of velocity vectors
    vector_scale : float
        Scale factor for velocity vectors
    vector_width : float
        Width of velocity vectors
    vector_skip : int
        Skip factor for velocity vectors (to avoid overcrowding)
        
    Returns
    -------
    None
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    
    # Compute vorticity
    # Compute grid spacing
    dx = np.mean(np.diff(x[0, :]))
    dy = np.mean(np.diff(y[:, 0]))
    
    # Compute derivatives
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    
    # Compute curl
    vorticity = dv_dx - du_dy
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create color plot
    im = plt.pcolormesh(x, y, vorticity, cmap=colormap, shading='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar if requested
    if show_colorbar:
        plt.colorbar(im)
    
    # Add velocity vectors if requested
    if show_vectors:
        plt.quiver(x[::vector_skip, ::vector_skip], 
                  y[::vector_skip, ::vector_skip], 
                  u[::vector_skip, ::vector_skip], 
                  v[::vector_skip, ::vector_skip], 
                  color=vector_color, 
                  width=vector_width, 
                  scale=1.0/vector_scale)
    
    # Add title and adjust plot
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to {output_path}")
    
    plt.show()


def streamline_plot(
    results: Dict[str, Any],
    density: float = 1.0,
    color: str = 'k',
    background_image: Optional[np.ndarray] = None,
    title: str = 'Streamlines',
    output_path: Optional[str] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Create a streamline plot of the velocity field.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing PIV or BOS results
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
    dpi : int
        DPI for saved image
    figsize : Tuple[int, int]
        Figure size in inches
        
    Returns
    -------
    None
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Display background image if provided
    if background_image is not None:
        plt.imshow(background_image, cmap='gray')
        plt.colormap('gray')
    
    # Plot streamlines
    plt.streamplot(x[0, :], y[:, 0], u, v, density=density, color=color)
    
    # Add title and adjust plot
    plt.title(title, fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    
    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
        print(f"Plot saved to {output_path}")
    
    plt.show()
