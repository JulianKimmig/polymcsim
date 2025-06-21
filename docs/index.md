# Welcome to PolyMCsim

A high-performance Python library for computational chemists to generate polymer graph structures through Monte Carlo simulations. The library models polymerization reactions using monomers as nodes and chemical bonds as edges, enabling emergent generation of diverse polymer architectures.

## Features

- Monte Carlo simulation of polymer growth
- Numba-optimized performance
- JSON/Pydantic configuration
- Batch simulation capabilities
- Support for complex monomer structures
- Parallel processing for large-scale simulations

## Installation

PolyMCsim requires Python 3.8 or later. You can install it using pip:

```bash
pip install polymcsim
```

Or with `uv`:

```bash
uv pip install polymcsim
```

## Basic Usage

Here is a basic example of how to generate a polymer:

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

For an interactive, step-by-step introduction, check out the [Getting Started](./tutorials/getting_started.md) tutorial.

### Installation

You can install `polymcsim` with pip:

```bash
pip install polymcsim
```

For more details on installation, including how to set up a development environment, see the [Installation Guide](./how-to/installation.md).

### Quick Example

Here’s a sneak peek at how `polymcsim` works. This example simulates the polymerization of a trifunctional monomer:

```python
from polymcsim import PolymerSimulation
import networkx as nx

# 1. Define monomers and reactions
// ... existing code ...
