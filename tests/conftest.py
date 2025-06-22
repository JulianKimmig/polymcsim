"""Test configuration and utilities for PolyMCsim."""

import os
import sys
from pathlib import Path
from typing import Generator, List, Union

import matplotlib
import matplotlib.pyplot as plt
import pytest
from pytest import FixtureRequest

# Use non-interactive backend for tests
matplotlib.use("Agg")

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Create a directory for test outputs
os.makedirs("test_output", exist_ok=True)


def setup_matplotlib() -> None:
    """Set up matplotlib for non-interactive testing."""
    matplotlib.use("Agg")


def verify_visualization_outputs(save_paths: List[Union[str, Path]]) -> None:
    """Verify that visualization files were created.

    Args:
        save_paths: List of file paths to check for existence.

    Raises:
        AssertionError: If any expected visualization file is missing.

    """
    for path in save_paths:
        assert os.path.exists(path), f"Expected visualization file {path} to be created"


@pytest.fixture(scope="function")
def plot_path(request: FixtureRequest) -> Generator[Path, None, None]:
    """Create a temporary directory for saving plots for a test session."""
    path = Path(__file__).parent / "test_output" / request.node.name
    path.mkdir(exist_ok=True, parents=True)

    yield path


@pytest.fixture
def get_plot_path(request: FixtureRequest, plot_path: Path) -> Path:
    """Create a subdirectory for a specific test function to save plots."""
    test_name = str(request.node.name)
    test_plot_path = plot_path / test_name
    test_plot_path.mkdir(exist_ok=True, parents=True)
    return test_plot_path


@pytest.fixture(autouse=True)
def cleanup_figures_after_test() -> Generator[None, None, None]:
    """Automatically close matplotlib figures after each test."""
    yield
    plt.close("all")


# Set up matplotlib for all tests
setup_matplotlib()
