#!/usr/bin/env python3
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio

# Run the MATLAB example
print("Running MATLAB BOS example...")
subprocess.run(["matlab", "-nodisplay", "-nosplash", "-r", 
                "try, cd('PIVsuite v.0.8.3'); example_09_BOS_image_pair; catch e, disp(getReport(e)); end; exit"], 
                check=True)

# Load the results
print("Loading results...")
# The MATLAB script should save the results in a .mat file
# We'll look for the most recent .mat file in the current directory
mat_files = [f for f in os.listdir("PIVsuite v.0.8.3") if f.endswith('.mat')]
if not mat_files:
    print("No .mat files found. The MATLAB script may not have saved the results.")
    exit(1)

# Sort by modification time (newest first)
mat_files.sort(key=lambda x: os.path.getmtime(os.path.join("PIVsuite v.0.8.3", x)), reverse=True)
latest_mat_file = os.path.join("PIVsuite v.0.8.3", mat_files[0])
print(f"Loading {latest_mat_file}")

# Load the .mat file
try:
    mat_data = sio.loadmat(latest_mat_file)
    print("Available variables in the .mat file:")
    for key in mat_data.keys():
        if not key.startswith('__'):  # Skip metadata
            print(f"  {key}: {type(mat_data[key])}")
except Exception as e:
    print(f"Error loading .mat file: {e}")
    exit(1)

# Load the original image
image_path = os.path.join("Data", "Test BOS", "11-49-28.000-4.tif")
try:
    image = np.array(Image.open(image_path))
    print(f"Image loaded: {image.shape}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Extract velocity vectors from the .mat file
# The exact variable names may vary, so we'll try common names
u = None
v = None
x = None
y = None

for key in mat_data.keys():
    if key.lower() == 'pivdata':
        pivdata = mat_data[key]
        # Try to extract u, v, x, y from pivdata
        if isinstance(pivdata, np.ndarray) and pivdata.dtype == np.dtype('O'):
            for field in pivdata.dtype.names:
                if field.lower() == 'u':
                    u = pivdata[field][0,0]
                elif field.lower() == 'v':
                    v = pivdata[field][0,0]
                elif field.lower() == 'x':
                    x = pivdata[field][0,0]
                elif field.lower() == 'y':
                    y = pivdata[field][0,0]

# If we couldn't find the vectors in pivdata, try other common variable names
if u is None and 'u' in mat_data:
    u = mat_data['u']
if v is None and 'v' in mat_data:
    v = mat_data['v']
if x is None and 'x' in mat_data:
    x = mat_data['x']
if y is None and 'y' in mat_data:
    y = mat_data['y']

# Check if we have all the data we need
if u is None or v is None:
    print("Could not find velocity vectors in the .mat file.")
    exit(1)
if x is None or y is None:
    print("Could not find coordinate grids in the .mat file.")
    # Create default coordinate grids
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]

print(f"Velocity field shape: u={u.shape}, v={v.shape}")
print(f"Coordinate grid shape: x={x.shape}, y={y.shape}")

# Create the plot
plt.figure(figsize=(12, 10))

# Display the image
plt.imshow(image, cmap='gray')

# Downsample the vectors for better visualization
step = 8  # Adjust this value to change the density of arrows
if len(u.shape) == 2:
    # 2D velocity field
    plt.quiver(x[::step, ::step], y[::step, ::step], 
               u[::step, ::step], v[::step, ::step], 
               color='r', scale=50, width=0.002)
else:
    # 1D arrays
    plt.quiver(x[::step], y[::step], u[::step], v[::step], 
               color='r', scale=50, width=0.002)

plt.title('BOS Image with Velocity Field')
plt.axis('off')
plt.tight_layout()

# Save the figure
output_path = 'bos_quiver_plot.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")

# Also save a version with just the quiver plot
plt.figure(figsize=(12, 10))
plt.quiver(x[::step, ::step], y[::step, ::step], 
           u[::step, ::step], v[::step, ::step], 
           color='k', scale=50, width=0.002)
plt.title('Velocity Field (Quiver Plot)')
plt.axis('equal')
plt.grid(True)
plt.tight_layout()

output_path = 'bos_quiver_only.png'
plt.savefig(output_path, dpi=300)
print(f"Quiver-only plot saved to {output_path}")

print("Done!")
