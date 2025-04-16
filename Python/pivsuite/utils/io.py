"""
I/O utility functions for PIVSuite Python

This module contains functions for loading and saving images.
"""

import numpy as np
from typing import Optional
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from a file path.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
        
    Returns
    -------
    np.ndarray
        Image as a numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine file extension
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()
    
    if ext in ['.tif', '.tiff']:
        # Use scikit-image for TIFF files
        from skimage import io
        img = io.imread(image_path)
    elif ext in ['.bmp', '.jpg', '.jpeg', '.png']:
        # Use PIL for common image formats
        from PIL import Image
        img = np.array(Image.open(image_path))
    elif ext in ['.mat']:
        # Use scipy for MATLAB files
        from scipy.io import loadmat
        mat = loadmat(image_path)
        # Try to find the image variable
        for key in mat.keys():
            if key not in ['__header__', '__version__', '__globals__'] and isinstance(mat[key], np.ndarray):
                img = mat[key]
                break
        else:
            raise ValueError(f"Could not find image data in MATLAB file: {image_path}")
    else:
        # Use scikit-image for other formats
        from skimage import io
        img = io.imread(image_path)
    
    # Convert to grayscale if RGB
    if len(img.shape) > 2:
        img = np.mean(img, axis=2).astype(np.float64)
    else:
        img = img.astype(np.float64)
    
    return img


def save_image(image: np.ndarray, image_path: str, format: Optional[str] = None) -> None:
    """
    Save an image to a file.
    
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array
    image_path : str
        Path to save the image
    format : Optional[str]
        Image format (e.g., 'tiff', 'png', 'jpg')
        
    Returns
    -------
    None
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(image_path)), exist_ok=True)
    
    # Determine file extension
    if format is None:
        _, ext = os.path.splitext(image_path)
        format = ext[1:].lower()
    
    # Normalize image to [0, 255] for common formats
    if format in ['png', 'jpg', 'jpeg', 'bmp']:
        image = np.clip(image, 0, None)
        image = 255 * image / np.max(image)
        image = image.astype(np.uint8)
    
    # Save image
    from skimage import io
    io.imsave(image_path, image)
