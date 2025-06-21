# Getting Started with PolyMCsim

This tutorial will guide you through the process of setting up and running a polymer simulation with `polymcsim`, and then visualizing the results. We will simulate the formation of a branched polymer and explore its properties.

## 1. Defining the System

First, you need to define the monomers and the reaction chemistry. Let's consider a simple system with a trifunctional monomer (`A3`) that can react with itself.

Here's how to define this system using `polymcsim`'s schema objects:

```python
from polymcsim import (
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

## 2. Setting Up the Simulation

Next, we define the simulation parameters, such as the number of monomers and the desired conversion.

```python
from polymcsim import Simulation

# ... (rest of the setup from above)
```

## 3. Running the Simulation

With the input defined, we can now run the simulation.

```python
from polymcsim import Simulation
# Assuming sim_input from the previous step is available

# Create and run the simulation
sim = Simulation(sim_input)
graph, metadata = sim.run()

print(f"Simulation completed with {metadata['reactions_completed']} reactions")
print(f"Final conversion: {metadata['final_conversion']:.1%}")
```

## 4. Visualizing the Results

`polymcsim` provides a suite of powerful visualization tools to analyze the simulation output. You'll need `matplotlib` installed to use them (`pip install matplotlib`).

### a. Polymer Structure

To visualize the structure of the largest polymer formed:

```python
from polymcsim import visualize_polymer
import matplotlib.pyplot as plt

fig = visualize_polymer(
    graph,
    component_index=0,  # Show largest polymer
    title="Largest Polymer Structure",
    save_path=output_dir / "polymer_structure.png"
)
plt.show()
```

### b. Molecular Weight Distribution

You can also plot the molecular weight distribution (MWD) of the polymers.

```python
from polymcsim import plot_molecular_weight_distribution
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

### c. Analysis Dashboard

For a comprehensive overview, you can create an analysis dashboard.

```python
from polymcsim import create_analysis_dashboard
import matplotlib.pyplot as plt

fig = create_analysis_dashboard(
    graph,
    metadata,
    title="Complete Polymer Analysis Dashboard",
    save_path=output_dir / "analysis_dashboard.png"
)
plt.show()
```

## 5. Exporting Data

Finally, you can export the polymer data to CSV files for further analysis.

```python
from polymcsim import export_polymer_data

export_files = export_polymer_data(
    graph,
    metadata,
    output_dir=output_dir,
    prefix="polymer_analysis"
)
print(f"Exported {len(export_files)} data files to {output_dir.resolve()}")
```

This will create `polymer_analysis_summary.csv` and `polymer_analysis_chain_data.csv`.

Congratulations! You've successfully run a simulation and used the basic visualization and data export capabilities of `polymcsim`.
