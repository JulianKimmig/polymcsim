# Getting Started with PolySim

This tutorial will guide you through the process of setting up and running a polymer simulation with `polysim`, and then visualizing the results. We will simulate the formation of a branched polymer and explore its properties.

## 1. Defining the Polymer System

First, we need to define the monomers and reactions for our simulation. We'll create a system with three types of monomers to produce a branched polymer.

- A **trifunctional monomer** that can act as a branching point.
- A **bifunctional monomer** to form linear chains.
- A **monofunctional monomer** that acts as a chain stopper, terminating growth.

Here's how to define this system using `polysim`'s schema objects:

```python title="create_system.py"
from polysim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    SimulationInput,
    SiteDef
)

def create_branched_polymer_system():
    """Create a branched polymer system for demonstration."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Trifunctional",
                count=30,
                molar_mass=150.0,
                sites=[SiteDef(type="A", status="ACTIVE")] * 3
            ),
            MonomerDef(
                name="Bifunctional",
                count=45,
                molar_mass=100.0,
                sites=[SiteDef(type="B", status="ACTIVE")] * 2
            ),
            MonomerDef(
                name="Monofunctional",  # Chain stopper
                count=10,
                molar_mass=50.0,
                sites=[SiteDef(type="B", status="ACTIVE")]
            ),
        ],
        reactions={
            frozenset(["A", "B"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=1.0
            )
        },
        params=SimParams(
            name="branched_polymer_demo",
            max_conversion=0.85,
            random_seed=42
        )
    )

# Get the configuration
sim_input = create_branched_polymer_system()
```

## 2. Running the Simulation

With the system defined, we can now run the simulation. We create a `Simulation` instance with our configuration and call the `run()` method.

```python title="run_simulation.py"
from polysim import Simulation
# Assuming sim_input from the previous step is available

# Create and run the simulation
sim = Simulation(sim_input)
graph, metadata = sim.run()

print(f"Simulation completed with {metadata['reactions_completed']} reactions")
print(f"Final conversion: {metadata['final_conversion']:.1%}")
```

The `run()` method returns a `networkx.Graph` object representing the polymer structures and a dictionary containing metadata about the simulation, such as the number of reactions and the final conversion.

## 3. Visualizing the Results

`polysim` provides a suite of powerful visualization tools to analyze the simulation output. You'll need `matplotlib` installed to use them (`pip install matplotlib`).

Let's create a directory to save our plots:

```python
from pathlib import Path

output_dir = Path("simulation_results")
output_dir.mkdir(exist_ok=True)
```

### Polymer Structure

You can visualize the structure of the largest polymer chain formed during the simulation.

```python title="visualize_structure.py"
from polysim import visualize_polymer
import matplotlib.pyplot as plt

fig = visualize_polymer(
    graph,
    component_index=0,  # Show largest polymer
    title="Largest Polymer Structure",
    save_path=output_dir / "polymer_structure.png"
)
plt.show()
```

### Molecular Weight Distribution

Analyzing the molecular weight distribution (MWD) is crucial. You can plot it on both linear and log scales.

```python title="visualize_mwd.py"
from polysim import plot_molecular_weight_distribution
import matplotlib.pyplot as plt

fig = plot_molecular_weight_distribution(
    graph,
    show_pdi=True,
    title="Molecular Weight Distribution",
    save_path=output_dir / "mwd_normal.png"
)
plt.show()

fig_log = plot_molecular_weight_distribution(
    graph,
    log_scale=True,
    show_pdi=True,
    title="Molecular Weight Distribution (Log Scale)",
    save_path=output_dir / "mwd_log.png"
)
plt.show()
```

### Analysis Dashboard

For a comprehensive overview, you can generate a single dashboard containing multiple analyses, including chain length distribution, conversion kinetics, and branching analysis.

```python title="create_dashboard.py"
from polysim import create_analysis_dashboard
import matplotlib.pyplot as plt

fig = create_analysis_dashboard(
    graph,
    metadata,
    title="Complete Polymer Analysis Dashboard",
    save_path=output_dir / "analysis_dashboard.png"
)
plt.show()
```

## 4. Exporting Data

Finally, you can export the raw polymer data to CSV files for further analysis in other tools.

```python title="export_data.py"
from polysim import export_polymer_data

export_files = export_polymer_data(
    graph,
    metadata,
    output_dir=output_dir,
    prefix="polymer_analysis"
)
print(f"Exported {len(export_files)} data files to {output_dir.resolve()}")
```

This concludes the getting started tutorial. You have learned how to define a polymer system, run a simulation, and use the visualization and data export capabilities of `polysim`.
