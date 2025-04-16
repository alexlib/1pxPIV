#!/usr/bin/env python3
"""
Command-line interface for PIVSuite Python.

This module provides a command-line interface for the PIVSuite package.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .core import analyze_image_pair, piv_params
from .visualization import quiver_plot, vector_plot


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="PIVSuite - Python implementation of PIVSuite for Particle Image Velocimetry"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # PIV analysis command
    piv_parser = subparsers.add_parser("piv", help="Perform PIV analysis on an image pair")
    piv_parser.add_argument("image1", help="Path to the first image")
    piv_parser.add_argument("image2", help="Path to the second image")
    piv_parser.add_argument("--output", "-o", help="Output directory for results", default="output")
    piv_parser.add_argument("--ia-size", help="Interrogation area size (comma-separated list)", default="32,16")
    piv_parser.add_argument("--ia-step", help="Interrogation area step (comma-separated list)", default="16,8")
    piv_parser.add_argument("--method", help="Interrogation method", choices=["basic", "offset", "defspline"], default="basic")
    
    # BOS analysis command
    bos_parser = subparsers.add_parser("bos", help="Perform BOS analysis on an image pair")
    bos_parser.add_argument("image1", help="Path to the first image")
    bos_parser.add_argument("image2", help="Path to the second image")
    bos_parser.add_argument("--output", "-o", help="Output directory for results", default="output")
    bos_parser.add_argument("--window-size", help="Window size for BOS analysis", type=int, default=32)
    bos_parser.add_argument("--step-size", help="Step size for BOS analysis", type=int, default=16)
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "piv":
        run_piv_analysis(args)
    elif args.command == "bos":
        run_bos_analysis(args)
    elif args.command == "version":
        from . import __version__
        print(f"PIVSuite version {__version__}")
    else:
        parser.print_help()
        sys.exit(1)


def run_piv_analysis(args):
    """Run PIV analysis on an image pair."""
    # Check if images exist
    if not os.path.exists(args.image1):
        print(f"Error: Image file not found: {args.image1}")
        sys.exit(1)
    
    if not os.path.exists(args.image2):
        print(f"Error: Image file not found: {args.image2}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse interrogation area parameters
    ia_size = [int(x) for x in args.ia_size.split(",")]
    ia_step = [int(x) for x in args.ia_step.split(",")]
    
    # Set PIV parameters
    piv_par = {}
    piv_par = piv_params(None, piv_par, 'defaults')
    
    # Customize parameters
    piv_par['ia_size_x'] = ia_size  # Interrogation area size in x
    piv_par['ia_size_y'] = ia_size  # Interrogation area size in y
    piv_par['ia_step_x'] = ia_step  # Interrogation area step in x
    piv_par['ia_step_y'] = ia_step  # Interrogation area step in y
    piv_par['ia_method'] = args.method  # Interrogation method
    
    print(f"Running PIV analysis on {args.image1} and {args.image2}...")
    print(f"Interrogation area size: {ia_size}")
    print(f"Interrogation area step: {ia_step}")
    print(f"Method: {args.method}")
    
    # Analyze image pair
    piv_data, _ = analyze_image_pair(args.image1, args.image2, None, piv_par)
    
    # Print some statistics
    print(f"Number of vectors: {piv_data['n']}")
    if 'masked_n' in piv_data:
        print(f"Number of masked vectors: {piv_data['masked_n']}")
    if 'spurious_n' in piv_data:
        print(f"Number of spurious vectors: {piv_data['spurious_n']}")
    
    # Create quiver plot
    print("Creating quiver plot...")
    quiver_plot(
        piv_data,
        scale=1.0,
        skip=2,
        color='r',
        title='PIV Velocity Field',
        output_path=str(output_dir / "piv_quiver_plot.png"),
        show=False
    )
    
    # Create vector plot of velocity magnitude
    print("Creating vector plot of velocity magnitude...")
    vector_plot(
        piv_data,
        component='magnitude',
        cmap='jet',
        title='PIV Velocity Magnitude',
        output_path=str(output_dir / "piv_velocity_magnitude.png"),
        show=False
    )
    
    print(f"Results saved to {output_dir}")


def run_bos_analysis(args):
    """Run BOS analysis on an image pair."""
    # Check if images exist
    if not os.path.exists(args.image1):
        print(f"Error: Image file not found: {args.image1}")
        sys.exit(1)
    
    if not os.path.exists(args.image2):
        print(f"Error: Image file not found: {args.image2}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Running BOS analysis on {args.image1} and {args.image2}...")
    print(f"Window size: {args.window_size}")
    print(f"Step size: {args.step_size}")
    
    # Import BOS module
    from .bos import analyze_bos_image_pair, plot_bos_results
    
    # Analyze image pair
    results = analyze_bos_image_pair(
        im1_path=args.image1,
        im2_path=args.image2,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    # Plot results
    print("Creating BOS visualization...")
    plot_bos_results(
        results=results,
        output_path=str(output_dir / "bos_results.png"),
        show=False
    )
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
