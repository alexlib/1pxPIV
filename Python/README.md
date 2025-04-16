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

### Standard PIV Analysis Example

```python
from pivsuite.core import analyze_image_pair, piv_params
from pivsuite.visualization import quiver_plot, vector_plot

# Set PIV parameters
piv_par = {}
piv_par = piv_params(None, piv_par, 'defaults')

# Customize parameters
piv_par['ia_size_x'] = [32, 16]  # Interrogation area size in x
piv_par['ia_size_y'] = [32, 16]  # Interrogation area size in y
piv_par['ia_step_x'] = [16, 8]   # Interrogation area step in x
piv_par['ia_step_y'] = [16, 8]   # Interrogation area step in y
piv_par['ia_method'] = 'defspline'  # Interrogation method ('basic', 'offset', 'defspline')

# Analyze image pair
piv_data, _ = analyze_image_pair(
    im1_path="path/to/first/image.tif",
    im2_path="path/to/second/image.tif",
    None,
    piv_par
)

# Create quiver plot
quiver_plot(
    piv_data,
    scale=1.0,
    skip=2,
    color='r',
    title='PIV Velocity Field',
    output_path="piv_quiver_plot.png"
)

# Create vector plot of velocity magnitude
vector_plot(
    piv_data,
    component='magnitude',
    cmap='jet',
    title='PIV Velocity Magnitude',
    output_path="piv_velocity_magnitude.png"
)
```

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
python examples/example_piv_analysis.py  # Standard PIV analysis
python examples/example_bos_image_pair.py  # BOS analysis
```

### Jupyter Notebooks

The package also includes Jupyter notebook versions of the examples in the `notebooks` directory. These notebooks provide an interactive way to learn and experiment with the PIVSuite Python package.

To run the notebooks, you need to have Jupyter installed:

```bash
pip install jupyter
```

Then you can start the Jupyter notebook server using the provided script:

```bash
python run_notebooks.py
```

Or manually:

```bash
cd notebooks
jupyter notebook
```

The notebooks include:

1. **example_01_image_pair_simple.ipynb** - Simple PIV analysis of an image pair
2. **example_02_image_pair_standard.ipynb** - Standard PIV analysis with custom parameters
3. **example_03_image_pair_advanced.ipynb** - Advanced PIV analysis with validation and smoothing
4. **example_04_piv_challenge_a4.ipynb** - Analysis of PIV Challenge A4 test case
5. **example_05_sequence_simple.ipynb** - Simple analysis of an image sequence
6. **example_06_sequence_fast_and_on_drive.ipynb** - Sequence analysis with saving results to disk
7. **example_07_piv_challenge_a3.ipynb** - Analysis of PIV Challenge A3 with multiple passes
8. **example_08a_piv_challenge_a1.ipynb** - Analysis of PIV Challenge A1
9. **example_08b_piv_challenge_a2.ipynb** - Analysis of PIV Challenge A2
10. **example_bos_cropped.ipynb** - Analysis of cropped BOS images
11. **example_bos_image_pair.ipynb** - Analysis of BOS image pairs
12. **example_piv_analysis.ipynb** - General PIV analysis

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
