# The Kinetic Monte Carlo Engine Behind PolySim

> “All models are wrong, but some are useful.” — George Box
>
> PolySim’s **Kinetic Monte Carlo (KMC)** core strikes a balance between physical realism and computational speed.

This article explains **how** the engine works and **why** certain design choices were made.  Understanding the internals will help you:

*   Pick the right simulation parameters.
*   Interpret the metadata returned by `Simulation.run()`.
*   Extend PolySim with new reaction types.

---

## 1 What Is Kinetic Monte Carlo?

KMC simulates **stochastic** time evolution of a chemical system by executing discrete reaction events with probabilities proportional to their **propensity** (rate × population size).

PolySim uses the classic **Gillespie algorithm** in its *direct* variant:

1.  Compute the total propensity \(a_0 = \sum_i a_i\).
2.  Draw two random numbers \(r_1, r_2 \in (0,1]\).
3.  Advance time by \(\Delta t = \frac{1}{a_0}\ln\left(\frac{1}{r_1}\right)\).
4.  Select reaction \(\mu\) such that \(\sum_{i<\mu} a_i < r_2 a_0 \le \sum_{i\le\mu} a_i\).
5.  Apply the reaction, update populations, and repeat.

PolySim implements this loop in **Numba-JIT-compiled** code for near-C speed.

---

## 2 Data Structures

| Structure | Purpose | Stored in |
|-----------|---------|-----------|
| `sites_data` | Lists every reactive site in the system with its monomer ID, site type, and status (`ACTIVE`, `DORMANT`, `CONSUMED`). | `numpy.ndarray` (shape \(N_\text{sites}\times 4\)) |
| `monomer_data` | Offsets into `sites_data` so the engine can quickly iterate over a monomer's sites. | `numpy.ndarray` |
| `available_sites_active` | For each site type holds indices of **active** sites that can react. | `numba.typed.Dict[int, List[int]]` |
| `available_sites_dormant` | Same as above but for **dormant** sites. | `numba.typed.Dict` |
| `site_position_map_*` | Maps a site's global index to its position in the corresponding list, enabling **O(1) swap-and-pop removal** after a reaction. | `numba.typed.Dict` |

!!! tip "Why typed dicts?"
    Python dictionaries are not natively supported inside Numba's `nopython` mode.  By using `numba.typed.Dict` with explicit key and value types we keep the inner loop fully JIT-compiled.

---

## 3 Handling Different Reaction Topologies

PolySim distinguishes three channel types:

1.  **Active–Active** (AA): both sites start `ACTIVE`.
2.  **Active–Dormant** (AD): an `ACTIVE` site reacts with a `DORMANT` site *and may activate* a hidden site afterwards (radical transfer).
3.  **Self-Reaction**: both reacting sites are of the **same type** (e.g., two radicals coupling).

All channels are stored in a fixed-length `numpy.ndarray` so that array operations (e.g.
propensity computation) vectorise well inside the JIT loop.

---

## 4 Activation Logic

A hallmark of PolySim is support for **post-reaction activation**—crucial for radical and step-growth systems.  The activation map in `ReactionSchema` tells the engine:

*   Which dormant site type becomes active.
*   What new active type it turns into.

After each reaction the engine scans the product monomer for the first matching dormant site **other than the one that just reacted**, updates its type, and moves it from the dormant to the active list.

---

## 5 Performance Tricks

*   **Swap-and-pop removal** keeps list updates O(1).
*   All random draws use `numpy.random` within Numba for speed.
*   Constant arrays are hoisted out of the time-critical loop.
*   The heavy function `run_kmc_loop` is wrapped with `@numba.njit(cache=True)` so it's compiled once and reused across runs.

On a modern laptop a 10-000-monomer simulation typically finishes in **seconds**, not minutes.

---

## 6 Limitations & Future Work

1.  **No spatial information.**  Diffusion or excluded-volume effects are ignored.
2.  **Constant rate coefficients.**  Temperature or conversion dependence would require user-supplied callback functions.
3.  **No explicit solvent.**  All reactions occur in a well-mixed bulk phase.

Pull requests tackling these limitations are welcome!

---

## 7 Key Takeaways

*   PolySim uses a Gillespie **direct KMC** algorithm compiled with Numba.
*   Smart data structures ensure O(1) updates, enabling large-scale simulations.
*   Activation logic allows modelling of complex branching and radical processes.

Understanding these internals empowers you to tune simulation parameters and interpret results with confidence. 