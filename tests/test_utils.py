import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests
import matplotlib.pyplot as plt

# Create a directory for test outputs
os.makedirs("test_output", exist_ok=True)


def setup_matplotlib():
    """Setup matplotlib for non-interactive testing."""
    matplotlib.use('Agg')


def cleanup_figure(fig):
    """Clean up a matplotlib figure to prevent memory leaks."""
    if fig is not None:
        plt.close(fig)


def verify_visualization_outputs(save_paths):
    """Verify that visualization files were created."""
    for path in save_paths:
        assert os.path.exists(path), f"Expected visualization file {path} to be created" 