# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A systems-biology modelling project around a single Boolean network of **autophagy / apoptosis
crosstalk** (MTOR–AMPK–ULK1–BECN1 upstream, caspase cascade downstream, phenotype readouts
`Apoptosis`, `Proliferation`, `BECN1`, `Autop_digestion`, `Senescence`). Everything in the repo is
either (a) a serialisation of that network, (b) a reusable Python analysis library, (c) a Jupyter
notebook running an analysis, or (d) a PhysiCell/PhysiBoSS multicellular simulation of the same
network. There is no build system, no test suite, and no packaging at the repo root — notebooks and
the `valley`/`maboss_visualize` packages are imported in place.

## Layout

| Path | Role |
|---|---|
| `model/` | The network itself, in every format it is needed in: `.zginml` (GINsim, the editable source of truth), `.bnet` (BoolNet/biobalm/mpbn), `.bnd` + `.bnd.cfg` (MaBoSS), `.sbml`, `.xgmml` (Cytoscape layout used for network figures). `convert_bnet_to_cana.py` converts `.bnet` → CANA truth-table format via sympy. |
| `valley/` | Importable library (not installed — imported from the repo root): attractor-landscape distances, clustering and embeddings for Boolean models. |
| `maboss_visualize/` | Importable library: plots MaBoSS node-activity on a Cytoscape-exported network layout, including animations. |
| `analysis/autophagy_model/`, `analysis/bcrn_boolean/` | Notebook analyses of the model (and of a BCRN-derived variant) using `maboss_visualize`. |
| `analysis/from_laurence/maboss_autophagy/` | The current CoLoMoTo notebook workflow (`Autophagy_Apoptosis_CoLoMoTo_Analysis_update.ipynb`) and the `valley` driver notebook (`Boolean_analysis.ipynb`). |
| `analysis/from_laurence/PhysiBoSS_autophagy/` | Self-contained PhysiCell + PhysiBoSS agent-based model (100 cells each running the Boolean network). Has its own `README.md` and `HOW_TO_RUN.md` — read those before touching it. |

Several notebooks in `analysis/autophagy_model/` and `analysis/bcrn_boolean/` still reference paths
from an older layout (`../boolean/…`, `../../model/bcrn_to_maboss/…`) that no longer exist. Fix the
path rather than assuming the model file is missing.

## The two model lineages — keep them straight

- `model/Autophagy_and_apoptosis.bnet` and `model/Autophagy_and_apoptosis_Sept26.bnet` each hold
  **119** rules. `Sept26` is the current revision and is what `Boolean_analysis.ipynb` loads.
- `analysis/from_laurence/PhysiBoSS_autophagy/config/boolean_network/autophagy_network.bnd` holds
  **125** nodes — a later revision, copied into the PhysiBoSS project and edited there. The two are
  *not* in sync; diff them before claiming a result from one applies to the other.
- The PhysiBoSS project also carries two locked-node mutant `.bnd`s
  (`..._BECN1_ON.bnd`, `..._rapamycin_MTOR_OFF.bnd`) produced by disconnecting a node from its
  regulators and forcing it ON/OFF.

## Shared scenario convention

Every analysis in the project — notebooks and PhysiBoSS alike — uses the same scenario definitions,
so a change to one must be mirrored in the others:

- **Base inputs**, all OFF except the constitutive two:
  `AA_Starvation=0, Growth_factor=0, TNF_R_or_DR=0, O_stress=0, Glucose_starv=0, Insuline=0,
  KEAP1=1, NQO1=1`.
- **Four scenarios**: growth-factor stimulation (`Growth_factor=1, Insuline=1`), amino-acid
  starvation (`AA_Starvation=1`), ER stress / glucose starvation (`Glucose_starv=1`), oxidative
  stress (`O_stress=1`).
- **Initialisation**: every node random (`set_istate(n, [0.5, 0.5])`), then inputs and the
  phenotype outputs pinned deterministically. The PhysiBoSS `.cfg` pins 18 nodes rather than 13
  (5 extra caspase-cascade nodes held OFF at t=0, deliberately — see its README).

Transient `Apoptosis` flickers under random initialisation are expected and are *not* commitment to
death; any analysis that reads death off a single timepoint will overcount.

## Running the Python analyses

Notebooks need the CoLoMoTo stack (`maboss`, `biolqm`, `ginsim`, `mpbn`, `biobalm`,
`pystablemotifs`, `pyboolnet`, `boolsim`, `cana`) plus `networkx`, `pygraphviz`/`pydot`, `seaborn`,
`scikit-learn`. There is no environment file; the CoLoMoTo Docker/conda notebook image is the
intended environment. `graphviz_layout` in `maboss_visualize` needs the graphviz binaries present.

Both local packages are imported via path manipulation, not installation:

```python
import sys; sys.path.append('../')        # analysis/<dir>/*.ipynb  -> maboss_visualize
import os; os.chdir('../../')             # Boolean_analysis.ipynb  -> repo root, then `import valley`
```

`os.chdir` to the repo root means all paths in `Boolean_analysis.ipynb` are repo-root-relative
(`model/Autophagy_and_apoptosis_Sept26.bnet`, `figures/`); re-running that cell twice breaks them.

### `valley` pipeline

The library takes a `.bnet` model plus an **attractor table** (one row per attractor, one column
per node, 0/1 or mean activity for cyclic attractors) and compares candidate distance metrics on
the attractor landscape. Computing attractors is deliberately out of scope — `Boolean_analysis.ipynb`
gets them from `biobalm.SuccessionDiagram`.

Layered, each module depending only on the ones above it:
`bnet` (parse, signed adjacency) → `geometry` (weight matrix, Laplacian, heat kernel; eigendecomposed
once and reused) → `distances` (each function returns a square matrix) → `compare` (build all
candidate matrices, rank them) → `clustering` / `embedding` → `plotting`.

```python
report = valley.run_comparison("model/Autophagy_and_apoptosis_Sept26.bnet", attractors)
```

is the one-call wrapper; the notebook does the same steps explicitly because it inspects the
intermediate quality report (degenerate matrices, heat kernels that merely track Hamming,
redundant pairs at Spearman > 0.98) before picking a distance. Pass the **same `t` grid** to
`build_distance_matrices` and `sweep_t` or the chosen `heat_kernel:t=…` key will not exist in both.

### `maboss_visualize`

`Net_visualizer` loads a Cytoscape export (`load_network_cyjs` / `load_network_xgmml`, using
`model/Autophagy_and_apoptosis.xgmml` for the curated layout), then an activity table or MaBoSS
time series, and renders static plots, two-condition comparisons, or frame-by-frame GIFs
(`create_networkactivity_animation`, which writes `network_plots/<name>/frame_*.png` next to the GIF).

## PhysiBoSS simulation

Build and run from `analysis/from_laurence/PhysiBoSS_autophagy/`:

```bash
PHYSICELL_CPP=/opt/homebrew/bin/g++-14 make -j4   # needs OpenMP; Apple clang will not work
./autophagy_population config/2_amino_acid_starvation.xml
python3 plot_population_over_time.py output_runs/2_amino_acid_starvation 30
./open_in_studio.sh 2_amino_acid_starvation.xml   # STUDIO_DIR is hardcoded to another machine
```

- `MABOSS_MAX_NODES=128` in the `Makefile` (default 64 is too small for 125 nodes); changing it
  changes which static MaBoSS library is linked.
- The first `make` runs `addons/PhysiBoSS/setup_libmaboss.py` to download a prebuilt libMaBoSS; the
  checked-in one is the **x86_64** build, matching the Intel Homebrew toolchain used so far. Delete
  `addons/PhysiBoSS/MaBoSS/` to let it re-fetch for a native arm64 compiler.
- 12 configs = 4 scenarios × {WT, BECN1-ON, rapamycin/MTOR-OFF}. Run them **one at a time** — each
  uses all cores via OpenMP and concurrent runs can silently truncate. Each writes to its own
  `output_runs/<name>/`.
- `BioFVM/`, `core/`, `modules/`, `addons/` are vendored PhysiCell/PhysiBoSS — do not edit. The
  project's own code is `main.cpp`, `custom_modules/custom.{cpp,h}`, `config/`, `Makefile`.
- Death is **not** wired through PhysiBoSS's output-mapping rate mechanism. `update_death_commitment()`
  in `custom.cpp` gates `Death::trigger_death()` on `Apoptosis` having been continuously ON for
  `apoptosis_commit_duration` (900 min) and then replays the cycle-switch / secretion-shutdown /
  death-phase sequence PhysiCell's core loop normally performs. The 900 min value is calibrated
  against `max_time=2880`; re-validate the 12-condition table in `README.md` if `max_time` changes.
- Cell colouring (`my_coloring_function`) is a fixed 5-way priority scheme
  (Apoptosis > Senescence > Autophagy > Proliferation > other); the `node_to_visualize` user
  parameter does **not** affect it. `plot_population_over_time.py` reproduces exactly that priority
  order and additionally counts cells fully removed by necrotic lysis.
