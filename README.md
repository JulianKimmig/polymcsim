# PolyMCsim

A high-performance Python library for computational chemists to generate polymer graph structures through Monte Carlo simulations. The library models polymerization reactions using monomers as nodes and chemical bonds as edges, enabling emergent generation of diverse polymer architectures.

## Features

- Monte Carlo simulation of polymer growth
- Numba-optimized performance
- JSON/Pydantic configuration
- Batch simulation capabilities
- Support for complex monomer structures
- Parallel processing for large-scale simulations

## Installation

PolyMCsim requires Python 3.8 or later. To install, run:

```bash
# Using Poetry (recommended)
poetry install

# Using pip
pip install .
```

## Development Setup

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd polymcsim
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Usage

Basic example of polymer generation:

```python
from polymcsim import PolymerSimulation

# Configure simulation
sim = PolymerSimulation(
    monomers_config="path/to/monomers.json",
    n_steps=1000
)

# Run simulation
result = sim.run()

# Export results
result.export_graph("polymer.graphml")
```

## Testing

Run the test suite:

```bash
poetry run pytest
```

## License

[License Type] - See LICENSE file for details

## Contributing

Contributions are welcome! Please read our Contributing Guidelines for details on how to submit pull requests, report issues, and contribute to the project.

*   **High Performance:** Built with `Numba` for C-like speed in computationally intensive parts.
*   **Extensible:** Easily define new monomer types, reactions, and simulation parameters.
*   **Rich Visualization:** Generate insightful plots and analyses right out of the box.

For detailed information, visit the full documentation at [juliankimmig.github.io/polymcsim/](https://juliankimmig.github.io/polymcsim/).

## Installation

You can install `polymcsim` via pip:

```bash
pip install polymcsim
```

Alternatively, you can use `uv`:

```bash
uv pip install polymcsim
```

## Quick Start

Here's a minimal example of how to simulate the formation of a branched polymer and visualize its structure:

```python
from polymcsim import PolymerSimulation

# 1. Define monomers and reactions
monomers = [
    # ... existing code ...
    # 4. Run the simulation
    sim = PolymerSimulation(simulation_input)
    polymer_graph = sim.run()

    # 5. Visualize the largest polymer structure
    sim.visualize_polymer(polymer_graph)
```

This will produce an image of the largest polymer's network structure. `polymcsim` also offers more advanced visualizations, such as molecular weight distribution plots and comprehensive analysis dashboards.
