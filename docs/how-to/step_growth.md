# Simulate Step-Growth Polymerization

!!! note "Prerequisites"
    * You have completed the [Getting Started](../tutorials/getting_started.md) tutorial.
    * `matplotlib` is installed (`pip install matplotlib`).

Step-growth (also called condensation) polymerization describes the reaction of bifunctional **A** and **B** sites to form linear chains.  In `polysim` you model this by assigning every monomer two reactive sites—one **A** and one **B**—that consume each other.

This guide shows you how to:

1.  Define a typical A–B step-growth system.
2.  Run the simulation until a target conversion is reached.
3.  Inspect the molecular-weight distribution (MWD).
4.  Export the resulting polymer graph for external analysis.

---

## 1 Create the Simulation Input

```python title="step_growth_system.py" linenums="1"
from polysim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    SimulationInput,
    SiteDef,
)

# Bifunctional monomer with two complementary sites
monomer = MonomerDef(
    name="AB-Monomer",
    count=1_000,           # 1 000 monomer units
    molar_mass=200.0,      # g · mol⁻¹
    sites=[
        SiteDef(type="A", status="ACTIVE"),
        SiteDef(type="B", status="ACTIVE"),
    ],
)

# When an A meets a B both sites are consumed (no activation products)
reaction = ReactionSchema(


    rate=1.0,              # relative rate constant
)

config = SimulationInput(
    monomers=[monomer],
    reactions={frozenset(["A", "B"]): reaction},
    params=SimParams(
        name="step_growth_demo",
        max_conversion=0.95,  # stop at 95 % functional-group conversion
        random_seed=7,
    ),
)
```

---

## 2 Run the Simulation

```python title="run_step_growth.py" linenums="1"
from polysim import Simulation
from step_growth_system import config

sim = Simulation(config)
graph, meta = sim.run()
print(f"Reactions executed: {meta['reactions_completed']}")
print(f"Final conversion : {meta['final_conversion']:.1%}")
```

The **conversion control** in `SimParams` stops the simulation early, preventing you from wasting CPU cycles once the desired extent of reaction is achieved.

---

## 3 Analyse the Molecular-Weight Distribution

```python title="analyse_mwd.py" linenums="1"
from polysim import plot_molecular_weight_distribution
import matplotlib.pyplot as plt

fig = plot_molecular_weight_distribution(
    graph,
    show_pdi=True,
    title="MWD – Step-Growth Polymerization",
)
plt.show()
```

At high conversions step-growth systems generate a broad, often **log-normal** MWD with high polydispersity index (PDI ≈ 2).

---

## 4 Export the Polymer Graph

```python title="export_graph.py" linenums="1"
from pathlib import Path
from polysim import export_polymer_data

export_dir = Path("step_growth_output")
export_files = export_polymer_data(
    graph,
    meta,
    output_dir=export_dir,
    prefix="step_growth",
)
print("Exported:")
for f in export_files.values():
    print(f"  - {f.relative_to(export_dir.parent)}")
```

You now have CSV files containing node, edge, and summary data that can be loaded into Excel, Pandas, or your favourite plotting tool.

---

## Next Steps

*   Vary the `max_conversion` parameter to investigate the **Carothers equation** predictions.
*   Introduce a small amount of mono-functional chain-stoppers to study how they limit molar mass.
*   Use different `rate` constants for multiple reactions to explore **selectivity**.
