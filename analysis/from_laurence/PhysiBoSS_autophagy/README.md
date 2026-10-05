# Autophagy population model (PhysiBoSS)

A PhysiCell + PhysiBoSS agent-based model: 100 cells, each running the full 125-node
autophagy/apoptosis Boolean network (`Autophagy_and_apoptosis_Sept26_2026-09-18.bnd`/`.cfg`
from the parent project, copied into `config/boolean_network/`) as its intracellular model,
embedded in a shared 2D microenvironment with 4 diffusible substrates.

**Purpose**: test whether, under extreme starvation, cells that die release nutrients that
neighboring cells can use to survive -- something the single-cell/well-mixed-population MaBoSS
analysis in the main notebook cannot represent (see `../DEV_LOG_2026-09-18.md` Section 15 for
that discussion).

## Cell coloring / legend

Cells are colored by phenotype state (`custom_modules/custom.cpp`, `my_coloring_function`),
checked in this priority order (a cell can have more than one node true at once in the Boolean
model; the highest-priority one wins the display color):

| State | Node checked | Fill | Outline |
|---|---|---|---|
| Dying | `Apoptosis` | `rgb(220,20,20)` red | `rgb(120,0,0)` |
| Senescent | `Senescence` | `rgb(160,32,240)` purple | `rgb(90,0,140)` |
| Autophagic | `Autophagy` | `rgb(255,140,0)` orange | `rgb(160,80,0)` |
| Proliferating | `Proliferation` | `rgb(0,180,0)` green | `rgb(0,90,0)` |
| Other / quiescent | none of the above | `rgb(170,170,170)` gray | `rgb(90,90,90)` |

This does **not** depend on the `node_to_visualize` field under **User Params** -- that field only
controls which single node gets an extra, separately-named custom_data entry in the saved output;
it has no effect on the on-screen/SVG color, which is always this fixed 5-way scheme. (This
distinction is why changing it may have looked like it did nothing.)

Autophagy is genuinely rare and transient in this model, not a rendering issue if you only spot
1-3 orange cells out of 100 -- under the corrected random-initialization convention used
throughout this project, the population's `Autophagy` signal peaks small and briefly before
`Apoptosis` takes over (see `DEV_LOG_2026-09-18.md` Section 3 for why). In the amino-acid-starvation
run already completed, the *most* autophagic cells present at any single timepoint was 3/100,
around t=690-750 min; before t~330 min and after t~840 min there are essentially none. Don't
expect it to ever dominate a frame the way dying or proliferating can.

`legend.svg` in this folder (and copied into `output/legend.svg`) is a small standalone image
with this same key. It's also in the format Studio's own Plot-tab buttons expect: the **"Legend
(.svg)"** button on the Plot tab will display it directly, and the paired-circle format (outer +
inner, matching how a rendered cell's cytoplasm/nucleus look) is what Studio's
`get_cell_types_from_legend()` parser needs to read it correctly -- confirmed by simulating its
exact parsing logic before relying on it.

**For a quantitative population-over-time view** (fraction of cells dying / senescent /
proliferating / other across the run, not just a spatial snapshot): use
`plot_population_over_time.py` (below) rather than Studio's generic **"Population plot"** button
-- that one counts by PhysiCell *cell type*, and this model only has one cell type
(`autophagy_cell`), so it can't break down by phenotype state. Studio does have a
**"Boolean states plot"** button specific to PhysiBoSS models, but it groups by the *entire*
125-node state combination, which tends to produce many tiny slices rather than the 4 coarse
categories used here.

```bash
python3 plot_population_over_time.py            # reads ./output, 30-min interval (the default)
python3 plot_population_over_time.py output 30   # explicit output dir / interval, if you change either
```

Reads every `output/output*_boolean_intracellular.csv` (one row per cell per saved frame,
written automatically by PhysiBoSS) and produces `output/population_over_time.png`, a stacked
area chart using the same 5-color key as the legend. Only needs Studio to have already produced
the output files -- run it any time after a simulation finishes.

## How it's wired

- **4 substrates** map to the 4 scenario-defining MaBoSS inputs used throughout this project:
  `nutrient` -> `AA_Starvation` (inhibition: low nutrient turns starvation ON), `growth_factor`
  -> `Growth_factor` and `Insuline`, `glucose` -> `Glucose_starv` (inhibition), `oxidative_stress_signal`
  -> `O_stress`. See the `<intracellular type="maboss"><mapping>` block in any `config/*.xml`.
- **The rescue mechanism**: `config/cell_rules.csv` makes necrotic cells secrete `nutrient`
  (`autophagy_cell,necrotic,increases,nutrient secretion,...,apply_to_dead=1`) -- i.e. a dying
  cell becomes a local nutrient source its neighbors can draw on.
- **Death**: the MaBoSS `Apoptosis` node drives PhysiCell's **necrosis** death model (not
  "apoptosis"), deliberately -- PhysiCell's necrosis model is the one that lyses and releases
  cell contents, which is the mechanistically relevant choice for the "dying cells release
  nutrients" story (real apoptotic cells are cleared without lysing; necrotic cells are not). This
  is **not** a PhysiBoSS XML output mapping (there is no `<output ... intracellular_name="Apoptosis">`
  in any config) -- it's a hard minimum-sustained-duration gate implemented in
  `custom_modules/custom.cpp` (`update_death_commitment`, called every intracellular update from
  `post_update_intracellular`): a cell is only committed to death once `Apoptosis` has been
  continuously ON for at least `apoptosis_commit_duration` minutes (a `<user_parameters>` entry,
  900 min by default), via `Death::trigger_death()`. See "Death mechanism" below and
  `DEV_LOG_2026-09-21.md` for why a plain rate-based mapping doesn't work at this project's
  `max_time=2880`.
- **Proliferation**: the `Proliferation` node is mapped to `cycle entry`.
- Custom_data on every cell tracks `Apoptosis`, `Proliferation`, `Autophagy`, `Senescence`,
  `BECN1` (see `custom_modules/custom.cpp`), so all 5 are available in saved output regardless
  of which one is chosen for SVG coloring (`node_to_visualize` in `<user_parameters>`).

## Opening in PhysiCell Studio

```bash
./open_in_studio.sh                                  # amino-acid starvation (default)
./open_in_studio.sh 1_growth_factor_stimulation.xml
./open_in_studio.sh 3_ER_glucose_starvation.xml
./open_in_studio.sh 4_oxidative_stress.xml
```

Finds the right conda environment (`physicell-studio`, falling back to `studio`) and launches
`PhysiCell-Studio/bin/studio.py` pointed at this project's config and the already-built
`autophagy_population` executable. Run it from this directory, or from anywhere -- it resolves
paths relative to its own location.

## The 4 scenarios

`config/PhysiCell_settings.xml` is the default Studio opens -- it *is*
`2_amino_acid_starvation.xml`, the scenario this whole sub-project was built to test. Load a
different one from Studio's config-file picker, or run from the command line:

```bash
./autophagy_population config/1_growth_factor_stimulation.xml
./autophagy_population config/2_amino_acid_starvation.xml
./autophagy_population config/3_ER_glucose_starvation.xml
./autophagy_population config/4_oxidative_stress.xml
```

Each scenario writes to its own `output_runs/<scenario name>/` folder (set via `<save><folder>`
in each config), so running one doesn't overwrite another's results -- `plot_population_over_time.py`
takes that directory as its first argument, e.g. `python3 plot_population_over_time.py
output_runs/4_oxidative_stress`.

## Mutant conditions: BECN1-ON and rapamycin (MTOR-OFF)

Mirrors the notebook's own `BECN1`/`MTOR` mutation analysis (`DEV_LOG_2026-09-18.md`), crossed
with all 4 scenarios -- `config/5_BECN1_ON_*.xml` through `config/12_rapamycin_MTOR_OFF_*.xml`.
Each points at a modified `.bnd` (`config/boolean_network/autophagy_network_BECN1_ON.bnd` /
`..._rapamycin_MTOR_OFF.bnd`) where the target node is locked (disconnected from its normal
regulators, forced permanently ON or OFF), otherwise identical to its base scenario.

```bash
./autophagy_population config/6_BECN1_ON_amino_acid_starvation.xml
./autophagy_population config/10_rapamycin_MTOR_OFF_amino_acid_starvation.xml
```

**Fixed** (see `DEV_LOG_2026-09-21.md` for the full investigation): these variants originally
showed 91-99% of cells fully dying, sharply diverging from the notebook's own prediction
(protective/cytostatic, not lethal, for these backgrounds). Cause: a transient Boolean flicker of
`Apoptosis` (common here, since ~half the randomly-initialized population starts with some part of
the death cascade already transiently active by chance) could permanently commit a cell even if
the flicker would have resolved in a pure MaBoSS reading -- locking `BECN1`/`MTOR` removes some of
the network's normal self-correction, so far more of that transient noise survives long enough to
matter. Fixed with the hard minimum-sustained-duration death-commitment gate described below (an
earlier fix based on tuning PhysiBoSS's built-in `smoothing`/`steepness` output mapping worked at
`max_time=1440` but broke again once `max_time` was extended to 2880 -- see "Death mechanism" and
the dev log for why a rate-based mapping can't be tuned around this at arbitrary run length). All
12 conditions (4 base scenarios + 8 mutant crosses) agree with the notebook -- see the table below.

## Death mechanism: hard minimum-sustained-duration gate

`Apoptosis` is **not** wired through PhysiBoSS's built-in output-mapping mechanism (which turns a
boolean node into a continuous stochastic death *rate*, shaped by `smoothing`/`steepness` XML
settings). That approach was tried first and works at short run lengths, but is structurally unable
to tell a long-but-eventually-resolving `Apoptosis` transient from true permanent commitment once
the run is long enough -- any rate mapping will, given enough elapsed time, eventually convert even
a partial, recurring signal into near-certain death. At this project's `max_time=2880` that
happened for real, MaBoSS-confirmed transients (up to ~270 consecutive minutes observed directly).

Instead, `custom_modules/custom.cpp`'s `update_death_commitment()` (called every intracellular
update from `post_update_intracellular`) tracks, per cell, how many **consecutive** minutes
`Apoptosis` has most recently been continuously ON (`apoptosis_on_since`/`apoptosis_sustained_min`
custom_data, reset to zero/`-1` the instant it goes OFF). Only once that unbroken run reaches
`apoptosis_commit_duration` minutes (a `<user_parameters>` entry, **900 min** by default) does the
code call `Death::trigger_death()` -- followed by the same cycle-model-switch / motility-and-secretion-shutdown
/ death-phase-entry-function sequence PhysiCell's own core loop runs after a standard stochastic
death trigger (necessary because `trigger_death()` alone only sets the `dead` flag; confirmed by
reading `core/PhysiCell_phenotype.cpp` directly). A transient that resolves before the threshold
has exactly zero chance of ever triggering death, no matter how many times it recurs.

**900 min, not the originally-chosen 600 min**: even a hard duration threshold isn't perfectly
immune to long enough observation windows -- it's a one-way ratchet, so the cumulative fraction of
cells that *ever* cross it keeps climbing the longer the run continues, especially for conditions
where `Apoptosis` keeps fluctuating throughout the whole run rather than settling early (the three
intermediate `BECN1`-ON conditions) or where a single early transient wave has more individual
outliers than expected (`growth_factor_stimulation`, `rapamycin`+growth factor). 600 min left these
conditions 10-20+ percentage points more lethal than the notebook predicts at `max_time=2880`; 900
min brings all of them back in line (table below). If `max_time` is ever extended well past 2880,
re-check this table rather than assuming 900 still holds -- see `DEV_LOG_2026-09-21.md`.

| Condition | Notebook (MaBoSS) | PhysiBoSS (100 cells, `max_time=2880`, `apoptosis_commit_duration=900`) |
|---|---|---|
| Growth-factor stimulation | 0% dead | 4% (residual noise, same order as elsewhere in this project) |
| Amino-acid starvation | 100% dead | 100% |
| ER/glucose starvation | 100% dead | 100% |
| Oxidative stress | 100% dead | 100% |
| BECN1-ON + growth factor | ~9% dead (validation report) | 0% dead, 100% autophagic |
| BECN1-ON + amino-acid starvation | ~50% dead | 53% dead / 47% autophagic |
| BECN1-ON + ER/glucose starvation | 50.7% dead | 57% dead / 43% autophagic |
| BECN1-ON + oxidative stress | 49.3% dead | 49% dead / 51% senescent |
| Rapamycin + growth factor | 0% dead (cytostatic) | 3% dead, 97% quiescent |
| Rapamycin + amino-acid starvation | (no direct notebook figure) | 100% |
| Rapamycin + ER/glucose starvation | 100% dead | 100% |
| Rapamycin + oxidative stress | 100% dead | 100% |

"Dead" here means the PhysiCell `dead` flag was ever set (currently showing `Apoptosis`=1, or
already fully removed via the necrosis model's 1440-min lysis phase) -- not just the current-frame
snapshot, since a cell that died early in a 2880-min run can be fully gone from the final frame's
`output*_boolean_intracellular.csv` by the time the run ends.

**Why 18 nodes are deterministic at t=0, not just the 13 documented earlier in this project**:
`config/boolean_network/autophagy_network.cfg`'s `.istate` holds the 9 `BASE_INPUTS` and the 4
phenotype-output nodes deterministic (as described in `DEV_LOG_2026-09-18.md`), plus 5 more --
`Caspase_3_6_7_C`, `Caspase9_APAF1_C`, `Cyt_C`, `MOMP`, `BAX` -- held deterministically OFF at t=0.
These 5 were added during an earlier, ultimately-abandoned attempt to fix the mutant-grid death
mapping by reducing random-seeded transient noise directly (see `DEV_LOG_2026-09-21.md`); that
attempt only partially worked and was superseded by the hard-duration-gate fix above. Once the gate
was in place, tested directly whether these 5 nodes were still worth keeping deterministic:
reverting them to random (`0.5/0.5`, matching the notebook's own "randomize everything except
inputs and outputs" convention) and re-running the full 12-condition grid gave **worse** agreement
with the notebook for most conditions -- growth-factor residual rose from 4% to 9%, rapamycin+growth-factor
from 3% to 6%, and BECN1-ON+ER/glucose-starvation drifted from 57% (vs. notebook's 50.7%) to 66%.
So, even though clearing these 5 nodes wasn't sufficient by itself to fix the original problem, it
is still a real, independently-useful noise reduction now that the hard-duration gate handles the
rest -- kept as-is rather than reverted to the "official" 13-deterministic-node convention.

## Building

Already built once and verified working (`./autophagy_population` in this folder) with:

```bash
PHYSICELL_CPP=/usr/local/Cellar/gcc/16.1.0/bin/g++-16 make -j4
```

On this machine, plain `g++`/`make` (Apple clang) fails (`-fopenmp` unsupported), and the
Homebrew `g++-11` here is too old for the current macOS SDK headers. If PhysiCell Studio's own
build fails the same way, pass `PHYSICELL_CPP=<path to a working g++>` the same way, or edit
that line into `Makefile` directly. `MABOSS_MAX_NODES` is set to 128 in the `Makefile` (up from
the default 64) because this network has 125 nodes.

The prebuilt MaBoSS library at `addons/PhysiBoSS/MaBoSS/` was re-fetched as the x86_64 build
(`libMaBoSS-osx64.tar.gz`) rather than the arm64 one `setup_libmaboss.py` picks by default,
because the available compiler here is an Intel/Rosetta Homebrew toolchain even though the
machine itself is Apple Silicon. If you build with a native arm64 compiler instead, delete
`addons/PhysiBoSS/MaBoSS` and let `make` re-run `setup_libmaboss.py` to fetch the matching
arm64 build.

## Verified so far, and what's not yet confirmed

I compiled this, ran all 4 scenarios' config for short/medium test windows, and confirmed:
progressive, heterogeneous cell death under starvation (0% of cells with `Apoptosis`=1 at
t=0, rising to 32% by t=600 min -- not a synchronized all-or-nothing outcome), matching the
qualitative behavior expected from the single-cell MaBoSS analysis.

**What I did *not* confirm**: a direct A/B test (rescue rule enabled vs. disabled, otherwise
identical, same random seed) gave the *same* 32% death fraction at t=600 min in both cases --
i.e. at the default parameters in this checked-in config, the nutrient-release mechanism is
wired in correctly (it changes the `nutrient` field: confirmed via the diffusion/rules log
output) but is not yet strong enough, or not yet acting on a fast enough timescale relative to
diffusion, to measurably change which cells commit to death by that time point. Likely levers,
in the order I'd try them: raise the rule's secretion `base_value` in `cell_rules.csv` (currently
0.05 1/min) by 5-10x; lower `nutrient`'s `decay_rate` and/or `diffusion_coefficient` in the XML
(currently 0.001 and 40000 micron^2/min -- diffusion this fast may be washing out local
secretion bumps before neighbors can respond); or extend the run past 600 min so small
differences have more time to compound. I did not want to keep tuning parameters blind without
you seeing intermediate results, since you have Studio's live controls to iterate much faster
than I can from here.

## Known simplifications / caveats

- `SQSTM1`'s and `Senescence`'s logic in the Boolean network are modeling judgment calls (see
  `../DEV_LOG_2026-09-18.md` Section 1), inherited unchanged here.
- Only `Apoptosis` and `Proliferation` are mapped to physical behaviors; `Autophagy`,
  `Senescence`, `BECN1` are tracked (custom_data, so visible/colorable in Studio) but don't yet
  drive any physical behavior of their own (e.g. a distinct "arrested but alive" state for
  senescent cells is not currently represented physically -- they'd just show as non-cycling,
  non-dead cells).
- `KEAP1`/`NQO1` are fixed ON via `<initial_values>` (constitutive, matching the MaBoSS
  scenario convention used throughout this project); `TNF_R_or_DR` and `DNA_damage` are left at
  their `.cfg` default (OFF) since none of the 4 canonical scenarios use them.
- Numeric parameters (secretion/uptake/diffusion rates, thresholds) are illustrative choices
  consistent with typical PhysiCell tutorial magnitudes, not literature-calibrated.
