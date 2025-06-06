"""
PIVSuite - Python implementation of PIVSuite for Particle Image Velocimetry

This package provides tools for analyzing particle image velocimetry (PIV) data.
It includes implementations of standard PIV, single-pixel PIV, and optical flow algorithms.
"""

__version__ = "0.1.0"

# Import core modules
from .core import (
    analyze_image_pair,
    analyze_image_sequence,
    piv_params
)

# Import visualization modules
from .visualization import (
    quiver_plot,
    vector_plot,
    scalar_plot,
    streamline_plot
)

__all__ = [
    # Core modules
    "analyze_image_pair",
    "analyze_image_sequence",
    "piv_params",

    # Visualization modules
    "quiver_plot",
    "vector_plot",
    "scalar_plot",
    "streamline_plot",
]
