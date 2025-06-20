"""Test configuration and utilities for PolySim."""

import os
import sys
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import pytest

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


@pytest.fixture
def plot_path(request) -> Path:
    """Create a unique output directory for each test based on its name and location."""
    # Get test module name (e.g., 'test_enhanced_visualization')
    module_name = request.module.__name__.split('.')[-1]
    
    # Get test function name
    test_name = request.node.name
    
    # Handle parametrized tests by cleaning the test name
    # e.g., "test_func[param1]" -> "test_func_param1"
    test_name = test_name.replace('[', '_').replace(']', '').replace('-', '_')
    
    # Create path: test_output/module_name/test_name/
    output_dir = Path(__file__).parent / "test_output" / module_name / test_name
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir
    
# Set up matplotlib for all tests
setup_matplotlib() 