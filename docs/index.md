# Welcome to PolySim

A high-performance Python library for computational chemists to generate polymer graph structures through Monte Carlo simulations. The library models polymerization reactions using monomers as nodes and chemical bonds as edges, enabling emergent generation of diverse polymer architectures.

## Features

- Monte Carlo simulation of polymer growth
- Numba-optimized performance
- JSON/Pydantic configuration
- Batch simulation capabilities
- Support for complex monomer structures
- Parallel processing for large-scale simulations

## Installation

PolySim requires Python 3.8 or later. You can install it using pip:

```bash
pip install polysim
```

Or with `uv`:

```bash
uv pip install polysim
```

## Basic Usage

Here is a basic example of how to generate a polymer:

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