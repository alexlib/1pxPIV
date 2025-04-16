"""
Quiver plot functions for PIVSuite Python

This module contains functions for creating quiver plots of PIV results.
It corresponds to the pivQuiver.m function in the MATLAB PIVsuite.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple


def quiver_plot(
    piv_data: Dict[str, Any],
    scale: float = 1.0,
    skip: int = 1,
    color: str = 'b',
    background_image: Optional[np.ndarray] = None,
    background: Optional[str] = None,  # Add this parameter
    title: str = 'Velocity Field',
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    xlabel: Optional[str] = None,  # Add this parameter
    ylabel: Optional[str] = None,  # Add this parameter
    crop: Optional[List[int]] = None,  # Add this parameter
    **kwargs
) -> plt.Figure:
    """
    Create a quiver plot of PIV results.

    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    scale : float
        Scale factor for arrows
    skip : int
        Skip factor for arrows (to avoid overcrowding)
    color : str
        Color of arrows
    background_image : Optional[np.ndarray]
        Background image to display under the vectors
    background : Optional[str]
        Background type ('magnitude', 'u', 'v', or None)
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
    xlabel : Optional[str]
        X-axis label
    ylabel : Optional[str]
        Y-axis label
    crop : Optional[List[int]]
        Crop region [x_min, x_max, y_min, y_max]
    **kwargs
        Additional keyword arguments for quiver plot

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

    # Apply cropping if specified
    if crop is not None:
        x_min, x_max, y_min, y_max = crop
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        valid = valid & mask

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Handle background
    if background == 'magnitude':
        # Calculate velocity magnitude
        magnitude = np.sqrt(u**2 + v**2)
        # Plot as background
        extent = [x.min(), x.max(), y.max(), y.min()] if crop is None else [x_min, x_max, y_max, y_min]
        plt.imshow(magnitude, extent=extent, origin='upper', cmap='jet', aspect='equal')
        plt.colorbar(label='Velocity Magnitude')
    elif background == 'u':
        # Plot u component as background
        extent = [x.min(), x.max(), y.max(), y.min()] if crop is None else [x_min, x_max, y_max, y_min]
        plt.imshow(u, extent=extent, origin='upper', cmap='jet', aspect='equal')
        plt.colorbar(label='U Component')
    elif background == 'v':
        # Plot v component as background
        extent = [x.min(), x.max(), y.max(), y.min()] if crop is None else [x_min, x_max, y_max, y_min]
        plt.imshow(v, extent=extent, origin='upper', cmap='jet', aspect='equal')
        plt.colorbar(label='V Component')
    elif background_image is not None:
        # Display background image
        plt.imshow(background_image, cmap='gray', origin='upper')

    # Plot quiver
    if crop is None:
        # Use all valid vectors with skip
        plt.quiver(
            x[::skip, ::skip][valid[::skip, ::skip]],
            y[::skip, ::skip][valid[::skip, ::skip]],
            u[::skip, ::skip][valid[::skip, ::skip]],
            v[::skip, ::skip][valid[::skip, ::skip]],
            color=color,
            scale=1.0/scale,
            **kwargs
        )
    else:
        # Use only vectors within the crop region
        mask = valid & (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        plt.quiver(
            x[::skip, ::skip][mask[::skip, ::skip]],
            y[::skip, ::skip][mask[::skip, ::skip]],
            u[::skip, ::skip][mask[::skip, ::skip]],
            v[::skip, ::skip][mask[::skip, ::skip]],
            color=color,
            scale=1.0/scale,
            **kwargs
        )

    # Add title and labels
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Set axis limits if cropping
    if crop is not None:
        plt.xlim(x_min, x_max)
        plt.ylim(y_max, y_min)  # Reversed for image coordinates
    else:
        plt.axis('equal')

    plt.tight_layout()

    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)

    # Show if requested
    if show:
        plt.show()

    return fig


def colored_quiver_plot(
    piv_data: Dict[str, Any],
    scale: float = 1.0,
    skip: int = 1,
    cmap: str = 'jet',
    background_image: Optional[np.ndarray] = None,
    background: Optional[str] = None,
    title: str = 'Velocity Field',
    output_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    crop: Optional[List[int]] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a colored quiver plot of PIV results.

    Parameters
    ----------
    piv_data : Dict[str, Any]
        Dictionary containing PIV results
    scale : float
        Scale factor for arrows
    skip : int
        Skip factor for arrows (to avoid overcrowding)
    cmap : str
        Colormap for arrows
    background_image : Optional[np.ndarray]
        Background image to display under the vectors
    background : Optional[str]
        Background type ('magnitude', 'u', 'v', or None)
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
    xlabel : Optional[str]
        X-axis label
    ylabel : Optional[str]
        Y-axis label
    crop : Optional[List[int]]
        Crop region [x_min, x_max, y_min, y_max]
    **kwargs
        Additional keyword arguments for quiver plot

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

    # Apply cropping if specified
    if crop is not None:
        x_min, x_max, y_min, y_max = crop
        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        valid = valid & mask

    # Compute velocity magnitude
    magnitude = np.sqrt(u**2 + v**2)

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Handle background
    if background == 'magnitude':
        # Calculate velocity magnitude
        # Plot as background
        extent = [x.min(), x.max(), y.max(), y.min()] if crop is None else [x_min, x_max, y_max, y_min]
        plt.imshow(magnitude, extent=extent, origin='upper', cmap='jet', aspect='equal')
        plt.colorbar(label='Velocity Magnitude')
    elif background == 'u':
        # Plot u component as background
        extent = [x.min(), x.max(), y.max(), y.min()] if crop is None else [x_min, x_max, y_max, y_min]
        plt.imshow(u, extent=extent, origin='upper', cmap='jet', aspect='equal')
        plt.colorbar(label='U Component')
    elif background == 'v':
        # Plot v component as background
        extent = [x.min(), x.max(), y.max(), y.min()] if crop is None else [x_min, x_max, y_max, y_min]
        plt.imshow(v, extent=extent, origin='upper', cmap='jet', aspect='equal')
        plt.colorbar(label='V Component')
    elif background_image is not None:
        # Display background image
        plt.imshow(background_image, cmap='gray', origin='upper')

    # Plot quiver
    q = plt.quiver(
        x[::skip, ::skip][valid[::skip, ::skip]],
        y[::skip, ::skip][valid[::skip, ::skip]],
        u[::skip, ::skip][valid[::skip, ::skip]],
        v[::skip, ::skip][valid[::skip, ::skip]],
        magnitude[::skip, ::skip][valid[::skip, ::skip]],
        cmap=cmap,
        scale=1.0/scale,
        **kwargs
    )

    # Add colorbar
    plt.colorbar(q, label='Velocity Magnitude')

    # Add title and labels
    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    # Set axis limits if cropping
    if crop is not None:
        plt.xlim(x_min, x_max)
        plt.ylim(y_max, y_min)  # Reversed for image coordinates
    else:
        plt.axis('equal')

    plt.tight_layout()

    # Save if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)

    # Show if requested
    if show:
        plt.show()

    return fig
