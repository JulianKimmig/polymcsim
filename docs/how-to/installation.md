# Installation Guide

This guide covers how to install `polymcsim` for both regular use and for development.

## Standard Installation

For most users, the recommended way to install `polymcsim` is from PyPI using `pip` or `uv`.

### Using `pip`

To install the latest stable version of `polymcsim`, run the following command in your terminal:

```bash
pip install polymcsim
```

### Using `uv`

If you prefer using `uv`, you can install `polymcsim` with:

```bash
uv pip install polymcsim
```

## Developer Installation

If you want to contribute to `polymcsim` or need the latest (unreleased) version, you should install it from the source.

### 1. Clone the Repository

First, clone the project's repository from GitHub:

```bash
git clone https://github.com/JulianKimmig/polymcsim.git
cd polymcsim
```

### 2. Install Dependencies

Once you have the source code, you can install the package and its development dependencies using `uv`. The `--all-extras` flag ensures that all optional dependencies, including those required for testing and documentation, are installed.

```bash
uv sync --all-extras
```

This command creates an editable installation, meaning that any changes you make to the source code will be immediately reflected when you run the package.

### 3. Verify Installation

To ensure everything is set up correctly, you can run the test suite:

```bash
uv run pytest
```

If all tests pass, your development environment is ready.
