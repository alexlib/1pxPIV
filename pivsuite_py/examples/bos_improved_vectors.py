#!/usr/bin/env python3
"""
Script to create improved vector visualization for the optimized optical flow results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, exposure
from scipy import ndimage

# Add the parent directory to the path so we can import the pivsuite package
sys.path.append(str(Path(__file__).parent.parent))


def load_results():
    """Load the previously computed optical flow results."""
    # Define paths
    output_dir = Path(__file__).parent.parent / "output"

    # Load the original image
    data_dir = Path(__file__).parent.parent.parent / "Data" / "Test BOS Cropped"
    im1_path = str(data_dir / "11-49-28.000-4.tif")
    im1 = io.imread(im1_path)
    im1 = im1.astype(np.float64) / np.max(im1)

    # Try to load the results from a saved file
    try:
        results = np.load(output_dir / "bos_optimized_flow_results.npz")
        x = results['x']
        y = results['y']
        u = results['u']
        v = results['v']
        print("Loaded results from saved file.")
    except:
        # If the file doesn't exist, recompute the results
        print("Saved results not found. Recomputing optical flow...")

        # Load second image
        im2_path = str(data_dir / "11-49-28.000-6.tif")
        im2 = io.imread(im2_path)
        im2 = im2.astype(np.float64) / np.max(im2)

        # Import the optimized flow function
        from bos_optimized_flow import optimized_lucas_kanade_flow

        # Run optimized optical flow
        results = optimized_lucas_kanade_flow(
            im1, im2,
            window_size=12,
            step_size=6,
            gradient_filter_size=2,
            temporal_filter_size=1,
            min_eigenvalue=1e-5
        )

        # Extract results
        x = results['x']
        y = results['y']
        u = results['u']
        v = results['v']

        # Save the results for future use
        np.savez(output_dir / "bos_optimized_flow_results.npz", x=x, y=y, u=u, v=v)

    return x, y, u, v, im1


def main():
    """Create improved vector visualizations."""
    print("\nCreating improved vector visualizations...")

    # Load the results
    x, y, u, v, im1 = load_results()

    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # We already loaded the image in the load_results function

    # Compute magnitude
    magnitude = np.sqrt(u**2 + v**2)

    # Normalize magnitude for visualization
    magnitude_norm = exposure.rescale_intensity(
        magnitude,
        in_range=(np.percentile(magnitude, 1), np.percentile(magnitude, 99))
    )

    # Create figure for vector field with autoscaling
    plt.figure(figsize=(15, 10))

    # Display the image
    plt.imshow(im1, cmap='gray')

    # Downsample the vectors for better visualization
    step = 3

    # Plot the quiver with autoscaling
    plt.quiver(x[::step, ::step], y[::step, ::step],
              u[::step, ::step], v[::step, ::step],
              color='r', scale_units='xy', scale=0.1, width=0.002,
              headwidth=3, headlength=5, headaxislength=4.5)

    plt.title('Optimized Optical Flow Vectors (Autoscaled)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # Save the vector field figure
    output_path = str(output_dir / "bos_optimized_flow_vectors_autoscaled.png")
    plt.savefig(output_path, dpi=600)
    print(f"Autoscaled vector field saved to {output_path}")

    # Create figure for vector field with normalized vectors
    plt.figure(figsize=(15, 10))

    # Display the image
    plt.imshow(im1, cmap='gray')

    # Normalize the vectors to unit length
    u_norm = u / (magnitude + 1e-10)  # Add small value to avoid division by zero
    v_norm = v / (magnitude + 1e-10)

    # Scale the normalized vectors by the magnitude for better visualization
    scale_factor = 5
    u_scaled = u_norm * np.power(magnitude, 0.5) * scale_factor
    v_scaled = v_norm * np.power(magnitude, 0.5) * scale_factor

    # Plot the quiver with normalized vectors
    plt.quiver(x[::step, ::step], y[::step, ::step],
              u_scaled[::step, ::step], v_scaled[::step, ::step],
              color='r', scale_units='xy', scale=1, width=0.002,
              headwidth=3, headlength=5, headaxislength=4.5)

    plt.title('Optimized Optical Flow Vectors (Normalized)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # Save the vector field figure
    output_path = str(output_dir / "bos_optimized_flow_vectors_normalized.png")
    plt.savefig(output_path, dpi=600)
    print(f"Normalized vector field saved to {output_path}")

    # Create figure for vector field with color-coded magnitude
    plt.figure(figsize=(15, 10))

    # Display the image
    plt.imshow(im1, cmap='gray')

    # Create a colormap for the vectors based on magnitude
    cmap = plt.cm.jet
    colors = cmap(magnitude_norm[::step, ::step].flatten())

    # Plot the quiver with color-coded magnitude
    q = plt.quiver(x[::step, ::step], y[::step, ::step],
                  u[::step, ::step], v[::step, ::step],
                  color=colors, scale_units='xy', scale=0.1, width=0.002,
                  headwidth=3, headlength=5, headaxislength=4.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Normalized Magnitude', fontsize=12)

    plt.title('Optimized Optical Flow Vectors (Color-Coded Magnitude)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # Save the vector field figure
    output_path = str(output_dir / "bos_optimized_flow_vectors_colored.png")
    plt.savefig(output_path, dpi=600)
    print(f"Color-coded vector field saved to {output_path}")

    # Create figure for vector field overlaid on magnitude
    plt.figure(figsize=(15, 10))

    # Display the magnitude
    plt.imshow(magnitude_norm, cmap='gray')

    # Plot the quiver with autoscaling
    plt.quiver(x[::step, ::step], y[::step, ::step],
              u[::step, ::step], v[::step, ::step],
              color='r', scale_units='xy', scale=0.1, width=0.002,
              headwidth=3, headlength=5, headaxislength=4.5)

    plt.title('Vectors Overlaid on Magnitude', fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # Save the vector field figure
    output_path = str(output_dir / "bos_optimized_flow_vectors_on_magnitude.png")
    plt.savefig(output_path, dpi=600)
    print(f"Vectors on magnitude saved to {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
