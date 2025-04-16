from setuptools import setup, find_packages

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
    author="PIVSuite Team",
    author_email="info@pivsuite.org",
    description="Python implementation of PIVSuite for Particle Image Velocimetry",
    keywords="piv, particle image velocimetry, fluid dynamics, image processing",
    url="https://github.com/pivsuite/pivsuite_py",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
