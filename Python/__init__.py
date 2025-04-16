"""
PIVSuite Python package.

This package provides Python implementations of PIV and BOS analysis algorithms.
"""

from .version import __version__

# Import core modules
from .core import (
    analyze_image_pair,
    analyze_image_sequence,
    piv_params
)

# Import BOS modules
from .bos import (
    analyze_bos_image_pair,
    plot_bos_results,
    plot_bos_quiver_only
)

# Import visualization modules
from .visualization import (
    quiver_plot,
    vector_plot,
    scalar_plot,
    streamline_plot
)

__all__ = [
    # Version
    "__version__",
    
    # Core modules
    "analyze_image_pair",
    "analyze_image_sequence",
    "piv_params",
    
    # BOS modules
    "analyze_bos_image_pair",
    "plot_bos_results",
    "plot_bos_quiver_only",
    
    # Visualization modules
    "quiver_plot",
    "vector_plot",
    "scalar_plot",
    "streamline_plot",
]
