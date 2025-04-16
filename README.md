# PIVSuite Python

Python implementation of PIVSuite for Particle Image Velocimetry (PIV) and Background-Oriented Schlieren (BOS) analysis.

## Features

- Standard PIV analysis with multi-pass, window deformation
- Single-pixel PIV analysis
- Background-Oriented Schlieren (BOS) analysis
- Visualization tools for PIV and BOS results
- Command-line interface for common tasks

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/pivsuite_py.git
cd pivsuite_py

# Install with pip
pip install -e .
```

For more detailed installation instructions, see [INSTALL.md](INSTALL.md).

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

### Command-Line Interface

PIVSuite provides a command-line interface for common tasks:

```bash
# Show help
pivsuite --help

# Run PIV analysis on an image pair
pivsuite piv image1.tif image2.tif --output results

# Run BOS analysis on an image pair
pivsuite bos image1.tif image2.tif --output results
```

### Running the Examples

The package includes several example scripts in the `examples` directory:

```bash
cd pivsuite_py
python examples/example_piv_analysis.py  # Standard PIV analysis
python examples/example_bos_image_pair.py  # BOS analysis
```

## Testing

Run the tests with pytest:

```bash
python -m pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original MATLAB PIVSuite by [PIVSuite Team]
- Contributors to the Python implementation
