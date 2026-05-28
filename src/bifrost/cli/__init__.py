"""Bifröst Command Line Interface.

Provides commands for running demos, benchmarks, and processing audio.

Example:
    $ bifrost demo 1              # Run anti-phase demo
    $ bifrost demo all            # Run all demos
    $ bifrost bench attention     # Run attention benchmark
    $ bifrost process audio.wav   # Process audio file through pipeline
"""

__version__ = "0.1.0"

from .main import main

__all__ = ["main"]
