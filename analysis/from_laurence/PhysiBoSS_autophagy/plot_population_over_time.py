"""
Build a cell-state-over-time population plot directly from a completed run's
output/output*_boolean_intracellular.csv files (one row per cell per saved frame,
listing every node currently ON). More reliable than Studio's built-in "Boolean
states plot" (which groups by the *full* 125-node state combination, so with a
network this size it tends to produce many tiny distinct-state slices rather
than the 5 coarse phenotype categories used for the Plot tab's coloring/legend).

Categories and priority order match custom_modules/custom.cpp's my_coloring_function
exactly: dying (Apoptosis) > senescent (Senescence) > autophagic (Autophagy) >
proliferating (Proliferation) > other/quiescent (none of the above).

[2026-09-21] Also tracks "removed" cells: PhysiCell's necrosis death model fully lyses
and removes a cell ~1440 min after it commits to death (the death model's phase-2
duration in every config here). On a long enough run (e.g. max_time=2880), a cell that
died early can complete this and disappear from the boolean_intracellular.csv entirely --
simply counting rows per frame would silently undercount total deaths once that happens.
Removed count per frame = (cells present in the very first frame) - (cells present in
this frame); reported as its own category rather than folded into "dying", so it stays
clear how many cells are actively in the dying/lysing state right now vs. fully gone.

Usage: run after a simulation has finished, from this directory:
    python3 plot_population_over_time.py [output_dir] [interval_minutes]
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 30  # must match <SVG><interval> / <full_data><interval> in the run's config

files = sorted(glob.glob(os.path.join(OUT, "output*_boolean_intracellular.csv")))
if not files:
    raise SystemExit(f"No output*_boolean_intracellular.csv files found under {OUT!r}. Run a simulation first.")
print(f"found {len(files)} frames in {OUT}")

times, dying, senescent, autophagic, proliferating, other, removed = [], [], [], [], [], [], []
initial_count = None

for i, f in enumerate(files):
    counts = {"dying": 0, "senescent": 0, "autophagic": 0, "proliferating": 0, "other": 0}
    n_present = 0
    with open(f) as fh:
        next(fh)  # header: ID,state
        for line in fh:
            _, state = line.rstrip("\n").split(",", 1)
            nodes = set(state.split(" -- "))
            n_present += 1
            if "Apoptosis" in nodes:
                counts["dying"] += 1
            elif "Senescence" in nodes:
                counts["senescent"] += 1
            elif "Autophagy" in nodes:
                counts["autophagic"] += 1
            elif "Proliferation" in nodes:
                counts["proliferating"] += 1
            else:
                counts["other"] += 1
    if initial_count is None:
        initial_count = n_present
    times.append(i * INTERVAL)
    dying.append(counts["dying"])
    senescent.append(counts["senescent"])
    autophagic.append(counts["autophagic"])
    proliferating.append(counts["proliferating"])
    other.append(counts["other"])
    removed.append(initial_count - n_present)

fig, ax = plt.subplots(figsize=(8, 5))
ax.stackplot(
    times, other, proliferating, autophagic, senescent, dying, removed,
    labels=["Other/quiescent", "Proliferating", "Autophagic", "Senescent", "Dying", "Removed (fully lysed)"],
    colors=["#aaaaaa", "#00b400", "#ff8c00", "#a020f0", "#dc1414", "#5a1a1a"],
)
ax.set_xlabel("time (min)")
ax.set_ylabel("number of cells")
ax.set_title("Cell state over time")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(0, max(times))
ax.set_ylim(0, initial_count)
plt.tight_layout()

out_path = os.path.join(OUT, "population_over_time.png")
plt.savefig(out_path, dpi=150)
print("saved", out_path)
print("final frame counts:", {"dying": dying[-1], "senescent": senescent[-1], "autophagic": autophagic[-1],
                               "proliferating": proliferating[-1], "other": other[-1], "removed": removed[-1]})
