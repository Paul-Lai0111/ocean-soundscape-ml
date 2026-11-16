"""Interactive soundscape explorer.

Run:  streamlit run app/dashboard.py
Then pick an output directory produced by `python -m oceansound.pipeline ...`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oceansound.config import BANDS  # noqa: E402

st.set_page_config(page_title="Ocean Soundscape Explorer", layout="wide")
PALETTE = px.colors.qualitative.Set2


@st.cache_data
def load_run(out: str):
    out = Path(out)
    df = pd.read_csv(out / "features_clustered.csv", parse_dates=["start_time"])
    z = np.load(out / "spectra.npz", allow_pickle=True)
    # spectra.npz is saved before sorting; align it to features_clustered.csv by segment id
    pos = {sid: i for i, sid in enumerate(z["segment_ids"])}
    idx = np.array([pos[s] for s in df["segment_id"]])
    summary = json.loads((out / "summary.json").read_text())
    summ = pd.read_csv(out / "cluster_summary.csv", index_col=0)
    conf = out / "confusion_true_vs_cluster.csv"
    conf = pd.read_csv(conf, index_col=0) if conf.exists() else None
    return df, z["ltsa"][idx], z["thumbs"][idx].astype(np.float32), z["freqs"], summary, summ, conf


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("Ocean Soundscape Explorer")
runs = sorted(p.parent for p in (ROOT / "outputs").glob("*/summary.json"))
if not runs:
    st.info("No runs found. Try:\n\n`python -m oceansound.pipeline synthetic --out outputs/synthetic`")
    st.stop()
run = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
df, ltsa, thumbs, freqs, summary, summ, conf = load_run(str(run))
df["cluster_str"] = "C" + df["cluster"].astype(str)
cluster_names = sorted(df["cluster_str"].unique(), key=lambda s: int(s[1:]))
color_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cluster_names)}

st.sidebar.markdown(f"**Windows:** {summary['n_windows']} x {summary['seg_seconds']:g} s")
st.sidebar.markdown(f"**Clusters (k):** {summary['k']}")
st.sidebar.markdown(f"**Silhouette:** {summary['silhouette_by_k'][str(summary['k'])]:.3f}")
if "adjusted_rand_index" in summary:
    st.sidebar.markdown(f"**ARI vs truth:** {summary['adjusted_rand_index']:.3f}")
st.sidebar.caption(f"{summary['time_range'][0]} -> {summary['time_range'][1]}")

tab_time, tab_clusters, tab_inspect, tab_about = st.tabs(
    ["Soundscape over time", "Clusters", "Inspect a window", "About the features"])

# ------------------------------------------------------- soundscape tab -----
with tab_time:
    st.subheader("Long-term spectral average (LTSA)")
    st.caption("Each column is one analysed window; colour is power spectral density (dB/Hz). "
               "Levels are relative (dB re full scale) unless a calibration offset was applied.")
    x = df["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    lo, hi = np.percentile(ltsa, [2, 99.5])
    fig = go.Figure(go.Heatmap(z=ltsa.T, x=x, y=freqs, colorscale="Viridis", zmin=lo, zmax=hi,
                               colorbar=dict(title="dB/Hz")))
    ticks = list(x.iloc[:: max(1, len(x) // 10)])
    fig.update_yaxes(type="log", title="Frequency (Hz)")
    fig.update_xaxes(type="category", tickvals=ticks, tickangle=-30)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.caption("Cluster of each window, in the same time order as the LTSA")
    strip = go.Figure(go.Bar(x=x, y=[1] * len(df), marker_color=df["cluster_str"].map(color_map),
                             marker_line_width=0, customdata=df[["cluster_str", "segment_id"]],
                             hovertemplate="%{x}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>",
                             showlegend=False))
    for c in cluster_names:  # legend entries
        strip.add_trace(go.Bar(x=[None], y=[None], marker_color=color_map[c], name=c))
    strip.update_yaxes(visible=False)
    strip.update_xaxes(type="category", categoryorder="array", categoryarray=list(x), showticklabels=False)
    strip.update_layout(height=110, bargap=0, margin=dict(l=10, r=10, t=0, b=0),
                        legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(strip, width="stretch")

    st.subheader("Band levels")
    band_cols = [c for c in BANDS if c in df.columns]
    picked = st.multiselect("Bands", band_cols, default=["b_10_100", "fin_17_25", "blue_40_50", "hump_200_2k"])
    if picked:
        long = df.melt(id_vars=["start_time"], value_vars=picked, var_name="band", value_name="dB")
        st.plotly_chart(px.line(long, x="start_time", y="dB", color="band", markers=True, height=350),
                        width="stretch")

# --------------------------------------------------------- clusters tab -----
with tab_clusters:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Windows in PCA space")
        fig = px.scatter(df, x="PC1", y="PC2", color="cluster_str", color_discrete_map=color_map,
                         category_orders={"cluster_str": cluster_names},
                         symbol="label" if df["label"].notna().all() else None,
                         hover_data=["segment_id", "start_time"], height=450)
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Model selection")
        sil = pd.Series({int(k): v for k, v in summary["silhouette_by_k"].items()}).sort_index()
        f2 = px.line(x=sil.index, y=sil.values, markers=True, labels={"x": "k", "y": "silhouette"}, height=250)
        f2.add_vline(x=summary["k"], line_dash="dash")
        st.plotly_chart(f2, width="stretch")
        ev = summary["pca_explained_variance"]
        st.caption("PCA explained variance: " + ", ".join(f"PC{i + 1} {v:.0%}" for i, v in enumerate(ev)))
        if conf is not None:
            st.subheader("True label vs cluster")
            st.dataframe(conf, width="stretch")

    st.subheader("Cluster profiles (feature means)")
    st.caption("Read each row as a fingerprint. High fin_excess means 20 Hz pulses; high low_minus_high_db "
               "and low flatness mean ship-like low-frequency noise; high var_200_2k means pulsed mid-frequency calls.")
    feat_cols = [c for c in summ.columns if c != "n_windows"]
    st.dataframe(summ.style.background_gradient(axis=0, cmap="RdBu_r", subset=feat_cols)
                 .format("{:.2f}", subset=feat_cols), width="stretch")

# ---------------------------------------------------------- inspect tab -----
with tab_inspect:
    c1, c2 = st.columns([1, 3])
    with c1:
        cl = st.selectbox("Cluster", cluster_names)
        ids = df.loc[df["cluster_str"] == cl, "segment_id"].tolist()
        sid = st.selectbox("Window", ids)
        row = df.index[df["segment_id"] == sid][0]
        st.markdown(f"**Start (UTC):** {df.loc[row, 'start_time']:%Y-%m-%d %H:%M:%S}  \n"
                    f"**Source:** {df.loc[row, 'source']}  \n"
                    f"**True label:** {df.loc[row, 'label'] if pd.notna(df.loc[row, 'label']) else 'n/a'}")
    with c2:
        T = thumbs[row]
        t = np.linspace(0, summary["seg_seconds"], T.shape[1])
        fig = go.Figure(go.Heatmap(z=T, x=t, y=freqs, colorscale="Magma", colorbar=dict(title="dB")))
        fig.update_yaxes(type="log", title="Frequency (Hz)")
        fig.update_xaxes(title="Time in window (s)")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title=sid)
        st.plotly_chart(fig, width="stretch")
    feats = summary["features"]
    comp = pd.DataFrame({"this window": df.loc[row, feats].astype(float),
                         "cluster mean": summ.loc[int(cl[1:]), feats].astype(float),
                         "all windows mean": df[feats].mean()}).round(2)
    st.dataframe(comp, width="stretch")

# ------------------------------------------------------------ about tab -----
with tab_about:
    st.markdown("### Frequency bands")
    st.table(pd.DataFrame([(k, f"{lo:g}-{hi:g} Hz", d) for k, (lo, hi, d) in BANDS.items()],
                          columns=["feature", "band", "what it targets"]))
    st.markdown("""
### Other features
- **fin_excess / blue_excess**: mean level in the call band minus its neighbouring bands (dB). Tonal or
  narrow-band calls stand out; broadband noise does not.
- **centroid_log10_hz**: power-weighted mean frequency (log10 Hz).
- **flatness_db**: spectral flatness; near 0 dB = noise-like and flat, very negative = peaky or steeply sloped.
- **low_minus_high_db**: 10-100 Hz level minus 1-8 kHz level.
- **var_10_100 / var_200_2k**: how much band energy fluctuates over 1 s frames (pulsed calls fluctuate).
- **p90_minus_med_200_2k**: how far loud frames rise above typical frames in 200-2000 Hz.
- **kurtosis**: impulsiveness of the waveform (0 for Gaussian noise).
""")
