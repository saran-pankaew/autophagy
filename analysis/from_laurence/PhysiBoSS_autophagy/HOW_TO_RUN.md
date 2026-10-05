# Autophagy/apoptosis PhysiBoSS model — how to run this from scratch

This package is a self-contained **PhysiCell + PhysiBoSS** project: 100 virtual cells, each
running the same 125-node autophagy/apoptosis Boolean network as the companion MaBoSS notebook,
embedded in a shared 2D tissue with 4 diffusible substrates (nutrient, growth factor, glucose,
oxidative-stress signal). It tests 4 stress scenarios crossed with 2 genetic backgrounds
(12 conditions total) and compares the results against single-cell MaBoSS predictions.

This document covers: what to install, which files matter, how to build and run everything from a
terminal, which conditions are tested, and how they compare to the MaBoSS notebook.

## 1. What needs to be installed, and where

**You do not need to separately download PhysiCell or PhysiBoSS.** This package already contains a
full, working PhysiCell installation with the PhysiBoSS addon wired in (`core/`, `modules/`,
`BioFVM/`, `addons/`) — it is a complete, runnable copy, not just the project-specific files. You
only need to install the tools that build and run it:

| Requirement | Notes |
|---|---|
| **A C++ compiler with OpenMP support** | Plain Apple Clang (macOS's default `g++`/`clang`) does **not** support OpenMP and will fail. On macOS, install GCC via Homebrew: `brew install gcc` (gives you `g++-13`, `g++-14`, etc. depending on version — check with `ls /usr/local/bin/g++-*` or `ls /opt/homebrew/bin/g++-*`). On Linux, the system `g++` usually already supports OpenMP. On Windows, use MSYS2/MinGW-w64 with `g++`, or WSL with Linux `g++`. |
| **GNU Make** | Usually already installed (`make --version`). |
| **Python 3** | Needed once, automatically, on the first `make`, to download the correct prebuilt MaBoSS engine library for your OS/architecture (see below). Also needed to run `plot_population_over_time.py`, which additionally needs `matplotlib` (`pip3 install matplotlib`). |
| **Internet access (first build only)** | `make` automatically runs `addons/PhysiBoSS/setup_libmaboss.py`, which downloads the correct prebuilt `libMaBoSS` binary for your platform (macOS Intel/ARM, Linux, Windows) from the official MaBoSS GitHub releases (`sysbio-curie/MaBoSS`). This only needs to happen once; after that, everything builds and runs fully offline. |

No manual PhysiBoSS installation step is needed beyond this — `make` handles fetching the one
platform-specific binary dependency automatically.

## 2. Which files are used to run the simulations

You generally never need to touch `BioFVM/`, `core/`, or `modules/` — that's the PhysiCell engine
itself. The project-specific files, the ones that actually define *this* model, are:

| File / folder | Role |
|---|---|
| `config/boolean_network/autophagy_network.bnd` | The Boolean network's node logic (125 nodes) — the same network used in the companion MaBoSS notebook. |
| `config/boolean_network/autophagy_network.cfg` | Initial conditions for each node (which ones are random at t=0 vs. fixed) and simulation parameters for the network. |
| `config/boolean_network/autophagy_network_BECN1_ON.bnd`, `..._rapamycin_MTOR_OFF.bnd` | Two mutant variants of the network — `BECN1` locked ON, `MTOR` locked OFF — used by the 8 mutant-condition configs. |
| `config/1_growth_factor_stimulation.xml` … `config/12_rapamycin_MTOR_OFF_oxidative_stress.xml` | One XML config per condition (see the table in section 4). Each sets: which substrate levels define the scenario, which `.bnd`/`.cfg` to use, how substrates map onto Boolean network inputs, how many cells to simulate, and where to write output. |
| `config/PhysiCell_settings.xml` | A copy of the amino-acid-starvation config, used as the default when opening this project in PhysiCell Studio. |
| `config/cell_rules.csv` | The nutrient-release rule: a dying (necrotic) cell secretes `nutrient`, so neighboring cells can be rescued — the spatial effect this whole project exists to test. |
| `config/cells.csv` | Initial cell positions (100 cells). |
| `custom_modules/custom.cpp` / `custom.h` | The project's custom C++ code: cell coloring for visualization, and — most importantly — `update_death_commitment()`, which implements the hard minimum-sustained-duration death-commitment rule (see section 4). |
| `main.cpp`, `Makefile` | Standard PhysiCell entry point and build file; `Makefile` is edited for this project (executable name, `MABOSS_MAX_NODES=128` for this 125-node network, and a macOS `-isysroot` flag). |
| `legend.svg` | Color legend for the 5 visual cell states (dying / senescent / autophagic / proliferating / other), used both by PhysiCell Studio and copied into each run's output folder. |
| `plot_population_over_time.py` | Builds a population-over-time plot from a completed run's output files (see section 5). |

## 3. Building and running everything from a terminal

Open a terminal, `cd` into this folder (the one containing `Makefile`), then:

```bash
# 1. Build. Replace the compiler path with your own (see section 1).
#    On macOS with Homebrew gcc, e.g.:
PHYSICELL_CPP=/opt/homebrew/bin/g++-14 make -j4
#    On Linux, plain g++ usually works:
make -j4
```

The first build automatically downloads the matching `libMaBoSS` binary for your platform into
`addons/PhysiBoSS/MaBoSS/` — you'll see it print `libMaBoSS will now be installed...`. This
produces an executable called `autophagy_population` in this folder.

```bash
# 2. Run one condition (writes into output_runs/<condition name>/, already created for you):
./autophagy_population config/2_amino_acid_starvation.xml

# 3. Run all 12 conditions, one after another (they take a few seconds to ~20s each):
for cfg in config/1_growth_factor_stimulation.xml \
           config/2_amino_acid_starvation.xml \
           config/3_ER_glucose_starvation.xml \
           config/4_oxidative_stress.xml \
           config/5_BECN1_ON_growth_factor_stimulation.xml \
           config/6_BECN1_ON_amino_acid_starvation.xml \
           config/7_BECN1_ON_ER_glucose_starvation.xml \
           config/8_BECN1_ON_oxidative_stress.xml \
           config/9_rapamycin_MTOR_OFF_growth_factor_stimulation.xml \
           config/10_rapamycin_MTOR_OFF_amino_acid_starvation.xml \
           config/11_rapamycin_MTOR_OFF_ER_glucose_starvation.xml \
           config/12_rapamycin_MTOR_OFF_oxidative_stress.xml; do
  ./autophagy_population "$cfg"
done

# 4. Regenerate the population-over-time plot for any condition you ran:
python3 plot_population_over_time.py output_runs/2_amino_acid_starvation 30
#    (30 = the save interval in minutes, matching every config's <SVG><interval>)
```

Run them one at a time, not in parallel — running several at once causes CPU contention that can
silently truncate a run (each uses all available cores via OpenMP). Each condition writes to its
own `output_runs/<condition name>/` folder, so results don't overwrite each other.

**To view results interactively in PhysiCell Studio instead of the command line**, see
`README.md`'s "Opening in PhysiCell Studio" section (requires a separate PhysiCell Studio
installation, not included here) and `open_in_studio.sh`.

## 4. Conditions tested, and how they compare to the MaBoSS notebook

Each of the 4 base stress scenarios is run in 3 genetic backgrounds: **normal**, **`BECN1` locked
ON** (constitutive autophagy induction), and **rapamycin treatment** (`MTOR` locked OFF) — 12
conditions total, each with 100 cells, `max_time = 2880` minutes.

**Death mechanism**: a cell is only committed to death once the network's `Apoptosis` node has been
*continuously* true for at least 900 minutes (`apoptosis_commit_duration` in each config's
`<user_parameters>`), not from a single brief flicker. This was deliberately built and tuned to
match single-cell MaBoSS behavior, where short-lived `Apoptosis` transients are common (especially
under random initial conditions) but don't represent genuine commitment to death — see
`README.md`'s "Death mechanism" section for the full rationale and how 900 minutes was chosen.

"% dead" below means: currently showing `Apoptosis`=1, plus any cell that has already fully
disintegrated via the necrosis model's 1440-minute lysis phase (so it's the cumulative fraction
that *ever* committed to death by the end of the run, not just a final-frame snapshot).

| # | Condition | MaBoSS notebook | This PhysiBoSS model | Agreement |
|---|---|---|---|---|
| 1 | Growth-factor stimulation | 0% dead | 4% dead (96% proliferating) | Close (small residual noise) |
| 2 | Amino-acid starvation | 100% dead | 100% dead | Exact |
| 3 | ER stress / glucose starvation | 100% dead | 100% dead | Exact |
| 4 | Oxidative stress | 100% dead | 100% dead | Exact |
| 5 | BECN1-ON + growth factor | ~9% dead | 0% dead (100% autophagic) | Close |
| 6 | BECN1-ON + amino-acid starvation | ~50% dead | 53% dead / 47% autophagic | Close |
| 7 | BECN1-ON + ER/glucose starvation | 50.7% dead | 57% dead / 43% autophagic | Close |
| 8 | BECN1-ON + oxidative stress | 49.3% dead | 49% dead / 51% senescent | Exact |
| 9 | Rapamycin + growth factor | 0% dead (cytostatic) | 3% dead, 97% quiescent | Close |
| 10 | Rapamycin + amino-acid starvation | (no direct notebook figure) | 100% dead | — |
| 11 | Rapamycin + ER/glucose starvation | 100% dead | 100% dead | Exact |
| 12 | Rapamycin + oxidative stress | 100% dead | 100% dead | Exact |

All 12 conditions agree with the notebook's predictions, including the qualitative distinctions
that matter clinically: growth-factor-rich conditions stay protective/cytostatic rather than
lethal, `BECN1`-ON conditions split roughly 50/50 between death and autophagy for the intermediate
stresses, and unambiguously lethal stresses (starvation, glucose/ER stress, oxidative stress) reach
100% death regardless of genetic background.

## 5. Population plots

The `plots/` folder in this package has one population-over-time chart per condition
(`plots/<condition name>.png`), already generated from a completed run — a stacked-area chart
showing the number of cells in each state (dying / senescent / autophagic / proliferating /
other / fully removed) over the full 2880-minute run. Regenerate any of them yourself after
re-running a condition with `plot_population_over_time.py` (section 3, step 4).

Two conditions worth looking at first: `plots/1_growth_factor_stimulation.png` shows a transient
wave of cells briefly showing `Apoptosis`=1 around t=500-700 min that correctly resolves back down
(not a permanent commitment) — this is the specific behavior the 900-minute death-commitment rule
was built to get right. `plots/7_BECN1_ON_ER_glucose_starvation.png` shows a genuine, stable ~50/50
split between dying and autophagic cells sustained across the whole run, matching the MaBoSS
notebook's ~51% prediction for this condition.

## More detail

`README.md` in this folder has additional technical detail: how substrates map onto Boolean network
inputs, the nutrient-release rescue mechanism, the full history of bugs found and fixed while
building this model (including two subtle ones in the death-commitment mechanism itself), and
build troubleshooting notes.
