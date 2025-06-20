"""Test configuration and utilities for PolySim."""

import os
import sys
from typing import List

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for tests
matplotlib.use('Agg')

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Create a directory for test outputs
os.makedirs("test_output", exist_ok=True)


def setup_matplotlib() -> None:
    """Set up matplotlib for non-interactive testing."""
    matplotlib.use('Agg')


def cleanup_figure(fig: plt.Figure) -> None:
    """Clean up a matplotlib figure to prevent memory leaks.
    
    Args:
        fig: The matplotlib figure to close.
    """
    if fig is not None:
        plt.close(fig)


def verify_visualization_outputs(save_paths: List[str]) -> None:
    """Verify that visualization files were created.
    
    Args:
        save_paths: List of file paths to check for existence.
        
    Raises:
        AssertionError: If any expected visualization file is missing.
    """
    for path in save_paths:
        assert os.path.exists(path), f"Expected visualization file {path} to be created"


# Set up matplotlib for all tests
setup_matplotlib() 