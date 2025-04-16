# PIVSuite Python

A Python implementation of PIVSuite for Particle Image Velocimetry (PIV) and Background-Oriented Schlieren (BOS) analysis.

## Overview

PIVSuite Python is a Python package that provides tools for analyzing particle image velocimetry (PIV) data and background-oriented schlieren (BOS) images. It is a Python port of the MATLAB PIVSuite package.

The package includes implementations of:
- Standard PIV algorithms
- Single-pixel PIV algorithms
- Lucas-Kanade optical flow for BOS analysis
- Visualization tools for PIV and BOS results

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib
- scikit-image
- Numba
- Pillow
- tqdm

### Installing from source

```bash
git clone https://github.com/pivsuite/pivsuite_py.git
cd pivsuite_py
pip install -e .
```

## Usage

### BOS Analysis Example

```python
from pivsuite.bos import analyze_bos_image_pair, plot_bos_results

# Analyze a pair of BOS images
results = analyze_bos_image_pair(
    im1_path="path/to/first/image.tif",
    im2_path="path/to/second/image.tif",
    window_size=32,
    step_size=16,
    scale=0.25  # Process at 25% of original size for faster computation
)

# Plot the results
plot_bos_results(
    results=results,
    output_path="bos_quiver_plot.png",
    quiver_scale=15.0,
    arrow_width=2.0,
    arrow_headsize=1.0,
    show_background=True
)
```

### Running the Examples

The package includes several example scripts in the `examples` directory:

```bash
cd pivsuite_py
python examples/example_bos_image_pair.py
```

## Testing

To run the tests:

```bash
cd pivsuite_py
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The original MATLAB PIVSuite developers
- The PIV and BOS research community
