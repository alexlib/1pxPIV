"""
Background-Oriented Schlieren (BOS) module for PIVSuite

This module implements the BOS analysis techniques for visualizing density gradients in fluids.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, Optional, Any
from skimage import io
import os

from .utils import load_image


def lucas_kanade_flow(
    im1: Union[str, np.ndarray],
    im2: Union[str, np.ndarray],
    window_size: int = 32,
    step_size: int = 16,
    start_row: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate optical flow using Lucas-Kanade method.

    Parameters
    ----------
    im1 : Union[str, np.ndarray]
        First image (either a numpy array or a path to an image file)
    im2 : Union[str, np.ndarray]
        Second image (either a numpy array or a path to an image file)
    window_size : int
        Size of the window for optical flow calculation
    step_size : int
        Step size between windows
    start_row : Optional[int]
        Starting row for analysis (if None, analyze the entire image)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing optical flow results:
        - 'x': x-coordinates of vectors
        - 'y': y-coordinates of vectors
        - 'u': x-component of displacement field
        - 'v': y-component of displacement field
    """
    # Load images if they are file paths
    if isinstance(im1, str):
        im1 = load_image(im1)

    if isinstance(im2, str):
        im2 = load_image(im2)

    # Convert to double if needed
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)

    # Get image dimensions
    height, width = im1.shape

    # Define the starting row
    if start_row is None:
        start_row = 0

    # Create grid of points
    x = np.arange(window_size // 2, width - window_size // 2 + 1, step_size)
    y = np.arange(start_row + window_size // 2, height - window_size // 2 + 1, step_size)
    X, Y = np.meshgrid(x, y)

    # Initialize velocity vectors
    u = np.zeros_like(X, dtype=np.float64)
    v = np.zeros_like(Y, dtype=np.float64)

    # Calculate optical flow for each point
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_pos = X[i, j]
            y_pos = Y[i, j]

            # Extract windows
            win1 = im1[y_pos - window_size // 2:y_pos + window_size // 2,
                       x_pos - window_size // 2:x_pos + window_size // 2]
            win2 = im2[y_pos - window_size // 2:y_pos + window_size // 2,
                       x_pos - window_size // 2:x_pos + window_size // 2]

            # Calculate gradient
            Gx, Gy = np.gradient(win1)

            # Calculate temporal gradient
            Gt = win2 - win1

            # Reshape gradients to vectors
            Gx_flat = Gx.flatten()
            Gy_flat = Gy.flatten()
            Gt_flat = Gt.flatten()

            # Create A matrix
            A = np.column_stack((Gx_flat, Gy_flat))

            # Calculate flow using least squares
            # Use pseudo-inverse for stability
            AtA = A.T @ A
            if np.linalg.det(AtA) > 1e-10:  # Check if matrix is invertible
                flow = -np.linalg.inv(AtA) @ A.T @ Gt_flat
                u[i, j] = flow[0]
                v[i, j] = flow[1]

    # Return results
    return {
        'x': X,
        'y': Y,
        'u': u,
        'v': v
    }


def analyze_bos_image_pair(
    im1_path: str,
    im2_path: str,
    window_size: int = 32,
    step_size: int = 16,
    scale: float = 1.0,
    start_row: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze a pair of BOS images to visualize density gradients.

    Parameters
    ----------
    im1_path : str
        Path to the first image
    im2_path : str
        Path to the second image
    window_size : int
        Size of the window for optical flow calculation
    step_size : int
        Step size between windows
    scale : float
        Scale factor for resizing images (1.0 = original size)
    start_row : Optional[int]
        Starting row for analysis (if None, analyze the entire image)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing BOS analysis results
    """
    print(f"Image paths:\n  {im1_path}\n  {im2_path}")

    # Check if images exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        raise FileNotFoundError("BOS image files not found. Please check the paths.")

    # Load images
    print("Loading images...")
    im1 = io.imread(im1_path)
    im2 = io.imread(im2_path)

    # Convert to grayscale if RGB
    if len(im1.shape) > 2:
        im1 = np.mean(im1, axis=2).astype(np.float64)
    else:
        im1 = im1.astype(np.float64)

    if len(im2.shape) > 2:
        im2 = np.mean(im2, axis=2).astype(np.float64)
    else:
        im2 = im2.astype(np.float64)

    # Resize images if needed
    if scale != 1.0:
        from skimage.transform import resize
        im1 = resize(im1, (int(im1.shape[0] * scale), int(im1.shape[1] * scale)),
                    anti_aliasing=True, preserve_range=True)
        im2 = resize(im2, (int(im2.shape[0] * scale), int(im2.shape[1] * scale)),
                    anti_aliasing=True, preserve_range=True)

    print(f"Images loaded and processed. Size: {im1.shape[1]} x {im1.shape[0]} pixels")

    # Calculate optical flow
    print("Calculating optical flow...")
    if start_row is not None:
        start_row = int(start_row * scale)
        print(f"Starting analysis from row {start_row} (scaled from original row {int(start_row/scale)})")

    flow_data = lucas_kanade_flow(im1, im2, window_size, step_size, start_row)

    # Store results
    results = {
        'im1': im1,
        'im2': im2,
        'x': flow_data['x'],
        'y': flow_data['y'],
        'u': flow_data['u'],
        'v': flow_data['v'],
        'window_size': window_size,
        'step_size': step_size,
        'scale': scale,
        'start_row': start_row
    }

    return results


def plot_bos_results(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    quiver_scale: float = 15.0,
    arrow_width: float = 2.0,
    arrow_headsize: float = 1.0,
    show_background: bool = True
) -> None:
    """
    Plot BOS analysis results.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing BOS analysis results
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    quiver_scale : float
        Scale factor for arrows in the quiver plot
    arrow_width : float
        Width of arrows in the quiver plot
    arrow_headsize : float
        Size of arrowheads in the quiver plot
    show_background : bool
        Whether to show the background image

    Returns
    -------
    None
    """
    # Extract data
    im1 = results['im1']
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']

    # Scale the vectors for better visualization
    u_scaled = u * quiver_scale
    v_scaled = v * quiver_scale

    # Create figure
    plt.figure(figsize=(12, 10))

    # Display the image if requested
    if show_background:
        plt.imshow(im1, cmap='gray')

    # Plot the quiver
    plt.quiver(x, y, u_scaled, v_scaled, color='r',
              linewidth=arrow_width, headwidth=arrow_headsize)

    # Add title and adjust the plot
    plt.title('BOS Image with Velocity Field', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()

    # Save the figure if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")

    plt.show()


def plot_bos_quiver_only(
    results: Dict[str, Any],
    output_path: Optional[str] = None,
    quiver_scale: float = 15.0,
    arrow_width: float = 2.0,
    arrow_headsize: float = 1.0
) -> None:
    """
    Plot only the quiver part of BOS analysis results.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing BOS analysis results
    output_path : Optional[str]
        Path to save the plot (if None, don't save)
    quiver_scale : float
        Scale factor for arrows in the quiver plot
    arrow_width : float
        Width of arrows in the quiver plot
    arrow_headsize : float
        Size of arrowheads in the quiver plot

    Returns
    -------
    None
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']

    # Scale the vectors for better visualization
    u_scaled = u * quiver_scale
    v_scaled = v * quiver_scale

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot the quiver
    plt.quiver(x, y, u_scaled, v_scaled, color='k',
              linewidth=arrow_width, headwidth=arrow_headsize)

    # Add title and adjust the plot
    plt.title('Velocity Field (Quiver Plot)', fontsize=16)
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure if requested
    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Quiver-only plot saved to {output_path}")

    plt.show()


def compute_bos_magnitude(results: Dict[str, Any]) -> np.ndarray:
    """
    Compute the magnitude of the BOS displacement field.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing BOS analysis results

    Returns
    -------
    np.ndarray
        Magnitude of the displacement field
    """
    # Extract data
    u = results['u']
    v = results['v']

    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)

    return magnitude


def compute_bos_divergence(results: Dict[str, Any]) -> np.ndarray:
    """
    Compute the divergence of the BOS displacement field.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing BOS analysis results

    Returns
    -------
    np.ndarray
        Divergence of the displacement field
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']

    # Compute grid spacing
    dx = np.mean(np.diff(x[0, :]))
    dy = np.mean(np.diff(y[:, 0]))

    # Compute derivatives
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)

    # Compute divergence
    divergence = du_dx + dv_dy

    return divergence


def compute_bos_curl(results: Dict[str, Any]) -> np.ndarray:
    """
    Compute the curl (vorticity) of the BOS displacement field.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing BOS analysis results

    Returns
    -------
    np.ndarray
        Curl of the displacement field
    """
    # Extract data
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']

    # Compute grid spacing
    dx = np.mean(np.diff(x[0, :]))
    dy = np.mean(np.diff(y[:, 0]))

    # Compute derivatives
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)

    # Compute curl
    curl = dv_dx - du_dy

    return curl
