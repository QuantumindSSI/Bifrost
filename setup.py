"""Setup configuration for spectral_encoder package."""

from setuptools import setup, find_packages

setup(
    name="spectral_encoder",
    version="0.1.0",
    description="Spectral encoder for FBC/QSSI: Audio/image/text ingestion & canonicalization",
    author="Quantumind Ltd",
    author_email="research@quantumind.ai",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "Pillow>=9.0.0",
        "soundfile>=0.11.0",
        "librosa>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
