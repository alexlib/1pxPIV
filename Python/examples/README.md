# PIVSuite Python Examples

This directory contains Python examples that match the Matlab examples in the `Matlab/html/` directory. These examples demonstrate how to use the PIVSuite Python package for various PIV analysis tasks.

## Prerequisites

Before running these examples, make sure you have:

1. Installed the PIVSuite Python package (see the main README.md for installation instructions)
2. Downloaded the required test data (see below)

## Test Data

The examples use test data that should be placed in the `Data/` directory at the root of the repository. Some examples use data from the PIV Challenge, which needs to be downloaded separately:

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

## Running the Examples

To run an example, use Python 3.8 or higher:

```bash
cd Python
python examples/example_01_image_pair_simple.py
```

## List of Examples

1. **example_01_image_pair_simple.py** - Simple PIV analysis of an image pair
   - Demonstrates the simplest possible use of PIVSuite Python for obtaining the velocity field from a pair of images

2. **example_02_image_pair_standard.py** - Standard PIV analysis with custom parameters
   - Shows how to customize PIV parameters for better results

3. **example_03_image_pair_advanced.py** - Advanced PIV analysis with validation and smoothing
   - Demonstrates advanced usage with custom validation, smoothing, and window functions

4. **example_04_piv_challenge_a4.py** - Analysis of PIV Challenge A4 test case
   - Treats images from test case A4 of 3rd PIV challenge by processing four quadrants separately

5. **example_05_sequence_simple.py** - Simple analysis of an image sequence
   - Shows how to process a sequence of PIV images and compute statistics

6. **example_06_sequence_fast_and_on_drive.py** - Sequence analysis with saving results to disk
   - Demonstrates how to save and load results to/from disk for faster reprocessing

7. **example_07_piv_challenge_a3.py** - Analysis of PIV Challenge A3 with multiple passes
   - Shows how to add iterations during processing and compare results

8. **example_08a_piv_challenge_a1.py** - Analysis of PIV Challenge A1
   - Demonstrates processing with multiple passes and computing statistics

9. **example_08b_piv_challenge_a2.py** - Analysis of PIV Challenge A2
   - Shows how to compute and visualize mean and RMS velocities, PDF, and spectra

## Output

The examples save their output (figures, etc.) to the `Python/output/` directory, which is created automatically if it doesn't exist.

## Comparison with Matlab Examples

These Python examples are designed to match the functionality and results of the corresponding Matlab examples in the `Matlab/html/` directory. The goal is to ensure that the Python code is fast, correct, and produces the same figures and results as the Matlab code.
