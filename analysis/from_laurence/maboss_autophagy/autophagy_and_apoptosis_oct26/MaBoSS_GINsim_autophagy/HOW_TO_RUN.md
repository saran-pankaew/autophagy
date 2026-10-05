# Autophagy/apoptosis MaBoSS + GINsim model -- how to run this from scratch

This package is the well-mixed (non-spatial) companion to the `PhysiBoSS_autophagy` package: a
125-node Boolean network describing autophagy/apoptosis crosstalk, together with a Jupyter
notebook analyzing it using the CoLoMoTo toolbox (GINsim, bioLQM, MaBoSS, mpbn).

**The current model is `Autophagy_and_apoptosis_Sept26_2026-09-23.bnd`/`.cfg`/`.zginml`.** The
`..._2026-09-18` and `..._2026-09-22` files are earlier network versions, kept only because the
notebook's own cells load them directly by filename for its historical scenario comparisons; they
are not the current model.

## 1. What needs to be installed, and where

The simplest path is the **CoLoMoTo Docker image**, which bundles everything this notebook needs
(GINsim, bioLQM, MaBoSS, mpbn, and their Python/Jupyter bindings) with no manual dependency
management:

```bash
docker pull colomoto/colomoto-docker
docker run -p 8888:8888 -v /path/to/this/folder:/notebook/MaBoSS_GINsim_autophagy colomoto/colomoto-docker
```

Then open the URL it prints (`http://127.0.0.1:8888/...`) and navigate into
`MaBoSS_GINsim_autophagy/`.

**Alternative: a local conda environment.**

```bash
conda create -n colomoto -c colomoto -c conda-forge colomoto-jupyter maboss ginsim biolqm mpbn python=3.11
conda activate colomoto
jupyter notebook
```

You also need a working **Java runtime** (GINsim and bioLQM run on the JVM under the hood via
`py4j`) -- the conda package above should pull one in automatically; if not, install a JDK (17+)
separately.

## 2. Which files are used, and how they fit together

| File | Role |
|---|---|
| `Autophagy_and_apoptosis_Sept26_2026-09-23.zginml` | **The current network**, in GINsim format. |
| `Autophagy_and_apoptosis_Sept26_2026-09-23.bnd`/`.cfg` | The same network, in plain-text MaBoSS format -- use these to run MaBoSS directly from the command line, inspect the network as text, or match against the companion `PhysiBoSS_autophagy` package (its `config/boolean_network/` files carry the identical network). |
| `Autophagy_Apoptosis_CoLoMoTo_Analysis_update_2026-09-22.ipynb` | The analysis notebook -- stable-state/attractor analysis, scenario/mutation screens, target screen. Its own cells load the `..._2026-09-18` and `..._2026-09-22` network files directly, which is why those two are also included; the notebook has not yet been re-run against the current, `..._2026-09-23` network. |
| `Autophagy_and_apoptosis_Sept26_2026-09-18.*`, `..._2026-09-22.*` | Earlier network versions, present only because the notebook references them by filename. Not the current model. |

## 3. Running MaBoSS directly on the current network

```python
import maboss
sim = maboss.load("Autophagy_and_apoptosis_Sept26_2026-09-23.bnd",
                   "Autophagy_and_apoptosis_Sept26_2026-09-23.cfg")
sim.update_parameters(max_time=60, sample_count=4000)
result = sim.run()
result.plot_trajectory()
```

To reproduce one of the 4 canonical stress scenarios, randomize every node except the
scenario-defining inputs, then fix those inputs:

```python
BASE_INPUTS = {"AA_Starvation": 0, "Growth_factor": 0, "TNF_R_or_DR": 0, "O_stress": 0,
               "Glucose_starv": 0, "Insuline": 0, "DNA_damage": 0, "KEAP1": 1, "NQO1": 1}
SCENARIOS = {
    "Growth-factor stimulation": {"Growth_factor": 1, "Insuline": 1},
    "Amino-acid starvation": {"AA_Starvation": 1},
    "ER stress / glucose starvation": {"Glucose_starv": 1},
    "Oxidative stress": {"O_stress": 1},
}
for n in sim.network:
    sim.network.set_istate(n, [0.5, 0.5])
inputs = dict(BASE_INPUTS); inputs.update(SCENARIOS["Amino-acid starvation"])
for node, val in inputs.items():
    sim.network.set_istate(node, [1 - val, val])
```

To test a `BECN1` mutation, add `sim.mutate("BECN1", "ON")` (or `"OFF"`) before running.

## 4. Results: the 12-condition comparison

Marginal probability of `Apoptosis` at `max_time=60`, `sample_count=4000`, for each of the 4
canonical scenarios crossed with 3 `BECN1` states:

| Scenario | WT | `BECN1`-ON | Rapamycin (`MTOR`-OFF) |
|---|---|---|---|
| Growth-factor stimulation | 0.000 | 0.000 | 0.000 |
| Amino-acid starvation | 1.000 | 0.510 | 1.000 |
| ER stress / glucose starvation | 1.000 | 0.491 | 1.000 |
| Oxidative stress | 1.000 | 0.504 | 1.000 |

Growth-factor stimulation is non-lethal in every background. `BECN1` locked ON provides strong,
consistent protection under every stress tested, roughly halving the marginal apoptosis
probability relative to WT/rapamycin. Rapamycin (`MTOR`-OFF) behaves like WT in this well-mixed
model under every scenario tested.

## 5. Note: the nutrient-rescue effect is PhysiBoSS-only

This notebook has no spatial dimension and no mechanism for one cell to affect another's state, so
it cannot represent the nutrient-rescue effect (dying cells locally raising nutrient levels to
spare not-yet-committed neighbors under amino-acid starvation) tested in the companion
`PhysiBoSS_autophagy` package. See that package's `README.md`/`HOW_TO_RUN.md` for those results.

## 6. Assumptions and limitations

- The network is curated from literature but not exhaustively validated node-by-node beyond the
  core apoptosis/autophagy decision nodes and the checks documented alongside this model.
- All nodes except the scenario-defining inputs are randomized at t=0 (standard MaBoSS
  convention); this notebook's own `.cfg` files hold every node deterministic OFF by default and
  rely on the notebook's own Python code to randomize them at simulation-build time.
- The oncogenic-background sweep and ~20-node target screen in the notebook have not been re-run
  against the current (`..._2026-09-23`) network.
