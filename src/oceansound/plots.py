"""Static figures (PNG) for README / slides. The interactive view is app/dashboard.py."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import BANDS  # noqa: E402

CMAP = "tab10"


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ltsa(ltsa: np.ndarray, freqs: np.ndarray, times: pd.Series, path: Path, title: str = "LTSA",
              units: str = "dB re FS"):
    """Long-term spectral average: one column per window, log-frequency axis."""
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(times))
    vmin, vmax = np.percentile(ltsa, [2, 99.5])
    m = ax.pcolormesh(x, freqs, ltsa.T, shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_yscale("log")
    ax.set_ylabel("Frequency (Hz)")
    step = max(1, len(times) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([pd.Timestamp(t).strftime("%m-%d %H:%M") for t in times.iloc[::step]],
                       rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    fig.colorbar(m, ax=ax, label=f"PSD ({units}/Hz)" if "uPa" not in units else "PSD (dB re 1 µPa²/Hz)")
    _save(fig, path)


def plot_band_levels(df: pd.DataFrame, path: Path, units: str = "dB re FS"):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(df))
    for name in ["b_10_100", "b_100_1k", "b_1k_8k", "fin_17_25", "blue_40_50"]:
        ax.plot(x, df[name], marker=".", lw=1, label=f"{name}  ({BANDS[name][0]:g}-{BANDS[name][1]:g} Hz)")
    ax.set_ylabel(f"Band level ({units})")
    ax.set_xlabel("Window index (time order)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Band levels over time")
    _save(fig, path)


def plot_pca(coords: np.ndarray, clusters: np.ndarray, path: Path, true_labels=None):
    ncols = 2 if true_labels is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    ax = axes[0, 0]
    cmap = plt.get_cmap(CMAP)
    ks = sorted(np.unique(clusters))
    names = {k: f"cluster {k}" for k in ks}
    colors = {k: cmap(k % 10) for k in ks}
    if true_labels is not None:
        # Colour each cluster like its majority true label so the panels match
        cats = pd.Categorical(true_labels)
        for k in ks:
            maj = pd.Series(np.asarray(true_labels)[clusters == k]).mode()[0]
            colors[k] = cmap(list(cats.categories).index(maj) % 10)
            names[k] = f"cluster {k} (mostly {maj})"
    for k in ks:
        m = clusters == k
        ax.scatter(coords[m, 0], coords[m, 1], color=colors[k], s=18, label=names[k])
    ax.set_title("PCA, coloured by cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    if true_labels is not None:
        ax = axes[0, 1]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=cats.codes, cmap=CMAP, s=18, vmin=0, vmax=9)
        ax.set_title("PCA, coloured by true label")
        ax.set_xlabel("PC1")
        handles, _ = sc.legend_elements()
        ax.legend(handles, list(cats.categories), fontsize=8)
    _save(fig, path)


def plot_silhouette(sil_by_k: dict[int, float], chosen_k: int, path: Path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ks = sorted(sil_by_k)
    ax.plot(ks, [sil_by_k[k] for k in ks], marker="o")
    ax.axvline(chosen_k, ls="--", c="grey")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"Model selection (chosen k = {chosen_k})")
    _save(fig, path)


def plot_cluster_examples(thumbs: np.ndarray, freqs: np.ndarray, clusters: np.ndarray,
                          seg_seconds: float, path: Path, per_cluster: int = 3, seed: int = 0):
    """A few example spectrogram thumbnails from each cluster."""
    rng = np.random.default_rng(seed)
    ks = sorted(np.unique(clusters))
    fig, axes = plt.subplots(len(ks), per_cluster, figsize=(3.4 * per_cluster, 2.2 * len(ks)),
                             squeeze=False, sharey=True)
    vmin, vmax = np.percentile(thumbs.astype(np.float32), [5, 99.5])
    for r, k in enumerate(ks):
        idx = np.flatnonzero(clusters == k)
        pick = rng.choice(idx, size=min(per_cluster, idx.size), replace=False)
        for c in range(per_cluster):
            ax = axes[r, c]
            if c < pick.size:
                T = thumbs[pick[c]].astype(np.float32)
                t = np.linspace(0, seg_seconds, T.shape[1])
                ax.pcolormesh(t, freqs, T, shading="nearest", cmap="magma", vmin=vmin, vmax=vmax)
                ax.set_yscale("log")
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(f"cluster {k}\nHz", fontsize=9)
            if r == len(ks) - 1:
                ax.set_xlabel("s")
    fig.suptitle("Example windows per cluster")
    _save(fig, path)


def plot_wind_vs_noise(df: pd.DataFrame, path: Path, feature: str = "b_1k_8k",
                       station: str = "46042", r: float | None = None, units: str = "dB re FS"):
    """Left: feature and buoy wind over time. Right: feature vs wind, coloured by cluster."""
    d = df.dropna(subset=["wind_ms"]).sort_values("start_time")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [2, 1]})
    t = pd.to_datetime(d["start_time"])
    a1.plot(t, d[feature], ".-", lw=0.8, ms=3, color="tab:blue", label=f"{feature} ({units})")
    a1.set_ylabel(f"{feature} ({units})", color="tab:blue")
    b1 = a1.twinx()
    b1.plot(t, d["wind_ms"], "-", lw=1.2, color="tab:orange", label="wind")
    b1.set_ylabel(f"Wind speed, NDBC {station} (m/s)", color="tab:orange")
    a1.set_title("High-frequency noise and wind over time")
    fig.autofmt_xdate()

    cmap = plt.get_cmap(CMAP)
    for i, k in enumerate(sorted(d["cluster"].unique())):
        s = d[d["cluster"] == k]
        a2.scatter(s["wind_ms"], s[feature], s=10, color=cmap(i % 10), label=f"cluster {k}")
    a2.set_xlabel(f"Wind speed, NDBC {station} (m/s)")
    a2.set_ylabel(f"{feature} ({units})")
    a2.set_title(f"Spearman r = {r:.2f}" if r is not None else "Noise vs wind")
    a2.legend(fontsize=8)
    _save(fig, path)


def plot_monthly(summary: pd.DataFrame, features: dict[str, str], path: Path, units: str = "dB re FS"):
    """One panel per feature: monthly median with interquartile band."""
    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.1 * n), sharex=True)
    x = np.arange(len(summary))
    for ax, (f, desc) in zip(np.atleast_1d(axes), features.items()):
        ax.fill_between(x, summary[f"{f}_q25"], summary[f"{f}_q75"], alpha=0.25, color="tab:blue")
        ax.plot(x, summary[f"{f}_median"], "o-", color="tab:blue", ms=4)
        unit = "dB" if "excess" in f or f.startswith("var_") else units
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(f"{f}: {desc}", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(summary.index, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Monthly median and interquartile range", fontsize=11)
    _save(fig, path)


def plot_month_hour(grids: dict[str, pd.DataFrame], path: Path):
    """Heatmaps of feature medians, month (rows) x local hour (columns)."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for ax, (name, g) in zip(axes[0], grids.items()):
        m = ax.imshow(g.to_numpy(), aspect="auto", cmap="viridis", origin="upper")
        ax.set_yticks(range(len(g.index)))
        ax.set_yticklabels(g.index, fontsize=7)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels(range(0, 24, 4))
        ax.set_xlabel("Local hour (Pacific time)")
        ax.set_title(name, fontsize=9)
        fig.colorbar(m, ax=ax, fraction=0.05)
    _save(fig, path)


def plot_index_comparison(monthly: pd.DataFrame, minute_df: pd.DataFrame, path: Path,
                          r_minute: dict[str, float], r_month: dict[str, float]):
    """Top: monthly curves of our feature vs MBARI CI. Bottom: per-minute scatter."""
    pairs = [("fin_excess", "mbari_fin_ci", "Fin whale"), ("blue_excess", "mbari_blue_ci", "Blue whale B call")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    x = np.arange(len(monthly))
    for j, (ours, ref, title) in enumerate(pairs):
        a = axes[0, j]
        a.plot(x, monthly[f"{ours}_median"], "o-", color="tab:blue", label=f"{ours} (median, dB)")
        a.set_ylabel(f"{ours} (dB)", color="tab:blue")
        b = a.twinx()
        b.plot(x, monthly[f"{ref}_mean"], "s--", color="tab:red", label=f"{ref} (mean)")
        b.set_ylabel("MBARI CI (1 = no signal)", color="tab:red")
        a.set_xticks(x)
        a.set_xticklabels(monthly.index, rotation=45, ha="right", fontsize=7)
        a.set_title(f"{title}: monthly, Spearman r = {r_month[ours]:.2f}", fontsize=10)

        s = axes[1, j]
        s.scatter(minute_df[ref], minute_df[ours], s=3, alpha=0.3, color="tab:gray")
        s.set_xlabel(f"{ref} (MBARI)")
        s.set_ylabel(f"{ours} (ours, dB)")
        s.set_title(f"Per minute, Spearman r = {r_minute[ours]:.2f} (n = {minute_df[[ours, ref]].dropna().shape[0]})",
                    fontsize=10)
    _save(fig, path)
