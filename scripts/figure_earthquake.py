"""Figure: the M3.3 Tres Pinos earthquake (2018-11-02 16:19:30 UTC) in the MARS recording.

Run after `python -m oceansound.pipeline mbari ... --out outputs/mbari_nov2018`.
Usage: python scripts/figure_earthquake.py [run_dir] [out_png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/mbari_nov2018")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/img/nov2018_earthquake.png")
ids = [f"MARS-20181102T000000Z-16kHz_{i:03d}" for i in (31, 32, 33)]

z = np.load(run / "spectra.npz", allow_pickle=True)
pos = {s: i for i, s in enumerate(z["segment_ids"])}
df = pd.read_csv(run / "features.csv", parse_dates=["start_time"]).set_index("segment_id")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, sid in zip(axes, ids):
    t = z["thumbs"][pos[sid]].astype(float)
    m = ax.pcolormesh(np.linspace(0, 60, t.shape[1]), z["freqs"], t, shading="nearest",
                      cmap="magma", vmin=-150, vmax=-60)
    ax.set_yscale("log")
    ax.set_xlabel("Time in window (s)")
    ax.set_title(f"{df.loc[sid, 'start_time']:%H:%M:%S} UTC   10-100 Hz: {df.loc[sid, 'b_10_100']:.0f} dB",
                 fontsize=9)
axes[0].set_ylabel("Frequency (Hz)")
axes[1].annotate("USGS: M3.3, 16:19:30 UTC\n12 km S of Tres Pinos, CA (~78 km)\narrives ~17 s after origin",
                 xy=(4, 25), xytext=(9, 1500), color="white", fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="white"))
fig.colorbar(m, ax=axes, label="dB re FS", fraction=0.025)
fig.suptitle("2018-11-02: three consecutive sampled minutes (30 min apart); the middle one catches an earthquake",
             fontsize=10)
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
