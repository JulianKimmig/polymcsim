# PolySim

A high-performance Python library for computational chemists to generate polymer graph structures through Monte Carlo simulations. The library models polymerization reactions using monomers as nodes and chemical bonds as edges, enabling emergent generation of diverse polymer architectures.

## Features

- Monte Carlo simulation of polymer growth
- Numba-optimized performance
- JSON/Pydantic configuration
- Batch simulation capabilities
- Support for complex monomer structures
- Parallel processing for large-scale simulations

## Installation

PolySim requires Python 3.8 or later. To install, run:

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
   cd polysim
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
from polysim import PolymerSimulation

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
