#!/usr/bin/env python3
"""
Setup script for PIVSuite Python package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pivsuite",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "numba>=0.54.0",
        "pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'flake8>=3.8.0',
            'black>=20.8b1',
            'mypy>=0.800',
        ],
        'test': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
        ],
    },
    author="PIVSuite Team",
    author_email="info@pivsuite.org",
    description="Python implementation of PIVSuite for Particle Image Velocimetry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="piv, particle image velocimetry, fluid dynamics, image processing",
    url="https://github.com/pivsuite/pivsuite_py",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pivsuite=pivsuite.cli:main',
        ],
    },
)
