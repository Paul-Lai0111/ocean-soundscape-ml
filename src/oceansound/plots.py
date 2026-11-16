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


def plot_ltsa(ltsa: np.ndarray, freqs: np.ndarray, times: pd.Series, path: Path, title: str = "LTSA"):
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
    fig.colorbar(m, ax=ax, label="PSD (dB/Hz)")
    _save(fig, path)


def plot_band_levels(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(df))
    for name in ["b_10_100", "b_100_1k", "b_1k_8k", "fin_17_25", "blue_40_50"]:
        ax.plot(x, df[name], marker=".", lw=1, label=f"{name}  ({BANDS[name][0]:g}-{BANDS[name][1]:g} Hz)")
    ax.set_ylabel("Band level (dB)")
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
