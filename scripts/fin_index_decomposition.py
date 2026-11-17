"""Why does fin_excess stay high in Jan-May while MBARI's fin call index falls?

Computes variants of a fin whale index from exact 1 Hz spectra, changing one
design choice at a time (peak band width, background frequencies, ratio vs
difference), and compares their monthly shapes after scaling each to
summer baseline = 0 and November = 1.

Finding: the background bands, not the peak width. Our 11-15 and 28-34 Hz
background contains the blue whale B call fundamental (~14 Hz) and 2nd harmonic
(~28 Hz), so blue whale calls in Aug-Nov depress fin_excess; MBARI's 12 / 34 Hz
avoid them. A cleaner background (11-13 and 33-36 Hz) fixes it.

Run scripts/low_freq_spectra.py first.
Usage: python scripts/fin_index_decomposition.py outputs/mbari_2018-19_hourly [docs/img/out.png]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

run = Path(sys.argv[1])
fig_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "figures" / "fin_index_decomposition.png"

z = np.load(run / "low_freq_spectra.npz", allow_pickle=True)
f, db = z["freqs"], z["psd_db"].astype(float)
lin = 10 ** (db / 10)
feats = pd.read_csv(run / "features.csv", parse_dates=["start_time"]).set_index("segment_id")
t = feats.loc[z["segment_ids"], "start_time"]
month = pd.to_datetime(t.values, utc=True).strftime("%Y-%m")


def bins(*hz):
    """Mean dB over single 1 Hz bins (MBARI style)."""
    return np.mean([db[:, np.where(f == h)[0][0]] for h in hz], axis=0)


def band(lo, hi):
    """dB of mean linear PSD over [lo, hi) Hz (pipeline style, as features.mean_psd_db)."""
    m = (f >= lo) & (f < hi)
    return 10 * np.log10(lin[:, m].mean(axis=1))


variants = {
    "MBARI CI (ratio)": bins(20, 21) / bins(12, 34),
    "MBARI bins, dB difference": bins(20, 21) - bins(12, 34),
    "narrow 20-21 Hz, our background": band(20, 22) - (band(11, 15) + band(28, 34)) / 2,
    "wide 17-25 Hz, MBARI background": band(17, 25) - bins(12, 34),
    "ours (fin_excess)": band(17, 25) - (band(11, 15) + band(28, 34)) / 2,
    "ours, clean background 11-13 & 33-36 Hz": band(17, 25) - (band(11, 13) + band(33, 36)) / 2,
}
V = pd.DataFrame(variants, index=z["segment_ids"])
V["month"] = month

# sanity checks against the pipeline's own columns
assert np.allclose(V["ours (fin_excess)"], feats.loc[V.index, "fin_excess"], atol=0.05)
assert np.allclose(V["MBARI CI (ratio)"], feats.loc[V.index, "mbari_fin_ci"], atol=1e-3)

monthly = V.groupby("month").mean()
base = monthly.loc[["2018-07", "2018-08"]].mean()
scaled = (monthly - base) / (monthly.loc["2018-11"] - base)
spring = scaled.loc["2019-01":"2019-05"].mean()

# Spectral shape: excess over the summer baseline at each 1 Hz bin, per month
summer = np.isin(month, ["2018-07", "2018-08"])
shape = pd.DataFrame({m: db[month == m].mean(axis=0) - db[summer].mean(axis=0) for m in sorted(set(month))},
                     index=f).T
peak_hz = shape.loc[:, 15:30].idxmax(axis=1)
sharp = shape.loc[:, 20:21].mean(axis=1) - shape.loc[:, [17, 18, 23, 24]].mean(axis=1)

pd.set_option("display.width", 200)
print("Monthly means scaled to summer = 0, November = 1:\n", scaled.round(2))
print("\nMean scaled value Jan-May (1 = as strong as November):\n", spring.round(2))
print("\nPeak frequency of excess (15-30 Hz) per month:\n", peak_hz.to_dict())
print("\nPeak sharpness: 20-21 Hz excess minus 17,18,23,24 Hz excess (dB):\n", sharp.round(2).to_dict())

# Contamination test: in Aug-Nov, does energy at the blue whale B harmonics track the blue index?
from scipy.stats import spearmanr  # noqa: E402

blue = feats.loc[V.index, "mbari_blue_ci"].to_numpy()
fin_ci = feats.loc[V.index, "mbari_fin_ci"].to_numpy()
season = pd.to_datetime(t.values, utc=True).month.isin([8, 9, 10, 11])
print("\nAug-Nov, Spearman r with MBARI blue CI:")
for name, x in [("28 Hz minus 34 Hz", bins(28) - bins(34)), ("14 Hz minus 12 Hz", bins(14) - bins(12))]:
    print(f"  {name}: {spearmanr(x[season], blue[season])[0]:+.2f}")
hi = season & (blue > np.quantile(blue[season], 0.8))
lo = season & (blue < np.quantile(blue[season], 0.2))
print("Aug-Nov, mean index in strongest minus weakest 20% blue whale minutes (dB):")
for name in ["ours (fin_excess)", "ours, clean background 11-13 & 33-36 Hz", "MBARI bins, dB difference"]:
    print(f"  {name}: {V.loc[hi, name].mean() - V.loc[lo, name].mean():+.2f}")
print(f"  (MBARI fin CI in those groups: {fin_ci[hi].mean():.3f} vs {fin_ci[lo].mean():.3f})")
print("\nPer-minute Spearman r with MBARI fin CI:")
for name in variants:
    print(f"  {name}: {spearmanr(V[name], fin_ci)[0]:.3f}")

out = run / "fin_index_decomposition.csv"
scaled.round(4).to_csv(out)
pd.DataFrame({"peak_hz": peak_hz, "sharpness_db": sharp.round(3)}).to_csv(run / "fin_peak_shape_by_month.csv")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
x = np.arange(len(scaled))
for name in variants:
    style = "--" if "clean" in name else "-"
    a1.plot(x, scaled[name], "o" + style, ms=3, label=name, lw=2 if "ours" in name or "CI" in name else 1)
a1.axhline(0, color="k", lw=0.5)
a1.set_xticks(x)
a1.set_xticklabels(scaled.index, rotation=45, ha="right", fontsize=7)
a1.set_ylabel("monthly mean, summer = 0, Nov = 1")
a1.set_title("Fin whale index variants, one design choice changed at a time", fontsize=10)
a1.legend(fontsize=7)
sel = shape.loc[:, 10:40]
m = a2.pcolormesh(sel.columns, np.arange(len(sel)), sel.values, cmap="RdBu_r", vmin=-10, vmax=10, shading="nearest")
a2.set_yticks(np.arange(len(sel)))
a2.set_yticklabels(sel.index, fontsize=7)
for hz in (12, 20, 21, 34):
    a2.axvline(hz, color="k", ls="--", lw=0.6)
a2.axvspan(17, 25, color="g", alpha=0.08)
for lo_, hi_ in ((11, 15), (28, 34)):
    a2.axvspan(lo_, hi_, color="orange", alpha=0.12)
a2.annotate("blue B\nfund.", (14, 1.2), fontsize=7, ha="center")
a2.annotate("blue B\n2nd harm.", (28, 1.2), fontsize=7, ha="center")
a2.set_xlabel("Hz; dashed = MBARI bins, green = our peak band, orange = our old background")
a2.set_title("Mean level minus Jul-Aug level (dB), exact 1 Hz bins", fontsize=10)
fig.colorbar(m, ax=a2, fraction=0.04)
fig.tight_layout()
fig.savefig(fig_path, dpi=130)
print("wrote", fig_path, out)
