# PIVSuite Python Jupyter Notebooks

This directory contains Jupyter notebook versions of the Python examples. These notebooks provide an interactive way to learn and experiment with the PIVSuite Python package.

## Prerequisites

Before running these notebooks, make sure you have:

1. Installed the PIVSuite Python package (see the main README.md for installation instructions)
2. Installed Jupyter: `pip install jupyter`
3. Downloaded the required test data (see below)

## Test Data

The notebooks use test data that should be placed in the `Data/` directory at the root of the repository. Some examples use data from the PIV Challenge, which needs to be downloaded separately:

- PIV Challenge A1: http://www.pivchallenge.org/pub05/A/A1.zip
- PIV Challenge A2: http://www.pivchallenge.org/pub05/A/A2.zip
- PIV Challenge A3: http://www.pivchallenge.org/pub05/A/A3.zip
- PIV Challenge A4: http://www.pivchallenge.org/pub05/A/A4.zip

After downloading, extract the files to the corresponding directories:
- `Data/Test PIVChallenge3A1/`
- `Data/Test PIVChallenge3A2/`
- `Data/Test PIVChallenge3A3/`
- `Data/Test PIVChallenge3A4/`

Other examples use data that should be included in the repository:
- `Data/Test von Karman/` - Flow around a cylinder
- `Data/Test Tububu/` - Turbulent flow in a channel
- `Data/Test BOS/` - Background-Oriented Schlieren images

## Running the Notebooks

To run the notebooks, start Jupyter:

```bash
cd Python
jupyter notebook notebooks/
```

Then select the notebook you want to run from the Jupyter interface.

## List of Notebooks

1. **example_01_image_pair_simple.ipynb** - Simple PIV analysis of an image pair
   - Demonstrates the simplest possible use of PIVSuite Python for obtaining the velocity field from a pair of images

2. **example_02_image_pair_standard.ipynb** - Standard PIV analysis with custom parameters
   - Shows how to customize PIV parameters for better results

3. **example_03_image_pair_advanced.ipynb** - Advanced PIV analysis with validation and smoothing
   - Demonstrates advanced usage with custom validation, smoothing, and window functions

4. **example_04_piv_challenge_a4.ipynb** - Analysis of PIV Challenge A4 test case
   - Treats images from test case A4 of 3rd PIV challenge by processing four quadrants separately

5. **example_05_sequence_simple.ipynb** - Simple analysis of an image sequence
   - Shows how to process a sequence of PIV images and compute statistics

6. **example_06_sequence_fast_and_on_drive.ipynb** - Sequence analysis with saving results to disk
   - Demonstrates how to save and load results to/from disk for faster reprocessing

7. **example_07_piv_challenge_a3.ipynb** - Analysis of PIV Challenge A3 with multiple passes
   - Shows how to add iterations during processing and compare results

8. **example_08a_piv_challenge_a1.ipynb** - Analysis of PIV Challenge A1
   - Demonstrates processing with multiple passes and computing statistics

9. **example_08b_piv_challenge_a2.ipynb** - Analysis of PIV Challenge A2
   - Shows how to compute and visualize mean and RMS velocities, PDF, and spectra

10. **example_bos_cropped.ipynb** - Analysis of cropped BOS images
    - Demonstrates BOS analysis on cropped images

11. **example_bos_image_pair.ipynb** - Analysis of BOS image pairs
    - Shows how to analyze BOS image pairs

12. **example_piv_analysis.ipynb** - General PIV analysis
    - Demonstrates general PIV analysis techniques

## Advantages of Jupyter Notebooks

- **Interactive**: Run code cells individually and see results immediately
- **Rich Output**: Visualize plots and images directly in the notebook
- **Documentation**: Combine code with rich text explanations
- **Experimentation**: Easily modify parameters and see the effects

## Comparison with Python Scripts

These notebooks contain the same functionality as the Python scripts in the `examples/` directory but in an interactive format. If you prefer to run the examples as scripts, you can use the Python files in the `examples/` directory instead.
