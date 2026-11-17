"""Ocean Soundscape: an interactive drawing set for one year of deep-ocean sound.

Run:  streamlit run app/dashboard.py
Lists the compact demo runs in demo/ (built by scripts/make_demo_data.py) and any run directory
produced by `python -m oceansound.pipeline ...` under outputs/.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(APP))
from oceansound.config import BANDS  # noqa: E402
from oceansound.data import file_offset_seconds, for_listening, read_window  # noqa: E402
import importlib  # noqa: E402

import theme as T  # noqa: E402

T = importlib.reload(T)  # a redeploy can keep the old module in memory; the theme is tiny

st.set_page_config(page_title="Ocean Soundscape · Paul Lai", layout="wide",
                   initial_sidebar_state="auto")
st.markdown(T.CSS, unsafe_allow_html=True)

REPO_URL = "https://github.com/Paul-Lai0111/ocean-soundscape-ml"
REPORT_URL = f"{REPO_URL}/blob/main/docs/analysis.md"
PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}
EARTHQUAKE_ID = "MARS-20181102T000000Z-16kHz_032"   # M3.3 Tres Pinos, 2018-11-02 16:19:30 UTC

SHEETS = {
    "G-0": ("Reading guide", "A six-step tour of this drawing set, and a handbook to keep beside it."),
    "S-1": ("General arrangement", "The whole record, and one minute of it you can hear."),
    "S-2": ("Cluster plan", "How the minutes group, and when each group occurs."),
    "S-3": ("Detail: one minute", "Spectrogram, sound and measured features of any minute."),
    "S-4": ("Validation and revisions", "Every interpretation checked against independent data."),
    "S-5": ("Notes and legend", "Features, bands, methods and sources."),
}

FEATURE_NAMES = {
    "broadband_db": "Broadband level, 10 Hz-7.9 kHz",
    "b_10_100": "Level 10-100 Hz (ships, large whales)",
    "b_100_1k": "Level 100 Hz-1 kHz",
    "b_1k_8k": "Level 1-7.9 kHz (wind, sea state)",
    "fin_17_25": "Fin whale band, 17-25 Hz",
    "blue_40_50": "Blue whale band, 40-50 Hz",
    "hump_200_2k": "Humpback band, 200 Hz-2 kHz",
    "fin_excess": "Fin whale excess over background",
    "blue_excess": "Blue whale excess over background",
    "centroid_log10_hz": "Spectral centroid (log10 Hz)",
    "flatness_db": "Spectral flatness",
    "low_minus_high_db": "Low minus high band",
    "var_10_100": "Fluctuation 10-100 Hz",
    "var_200_2k": "Fluctuation 200 Hz-2 kHz",
    "p90_minus_med_200_2k": "Loud frames above typical, 200 Hz-2 kHz",
    "kurtosis": "Impulsiveness (kurtosis)",
}


# ------------------------------------------------------------------ data -----
@st.cache_data
def load_run(out: str):
    out = Path(out)
    df = pd.read_csv(out / "features_clustered.csv", parse_dates=["start_time"])
    z = np.load(out / "spectra.npz", allow_pickle=True)
    pos = {sid: i for i, sid in enumerate(z["segment_ids"])}           # align spectra to the table
    idx = np.array([pos[s] for s in df["segment_id"]])
    summary = json.loads((out / "summary.json").read_text())
    summ = pd.read_csv(out / "cluster_summary.csv", index_col=0)
    conf = out / "confusion_true_vs_cluster.csv"
    conf = pd.read_csv(conf, index_col=0) if conf.exists() else None
    thumbs = z["thumbs"][idx].astype(np.float32)
    if "thumb_db_range" in z.files:                                      # 8-bit demo thumbnails
        lo, hi = z["thumb_db_range"]
        thumbs = lo + thumbs / 255.0 * (hi - lo)
    return df, z["ltsa"][idx].astype(np.float32), thumbs, z["freqs"], summary, summ, conf


@st.cache_data(show_spinner="Streaming one minute from the MBARI archive...", max_entries=32)
def load_audio(source: str, start_time: pd.Timestamp, seconds: float):
    offset = file_offset_seconds(source, start_time.to_pydatetime())
    return read_window(source, offset, seconds)


def run_title(path: Path) -> str:
    try:
        return json.loads((path / "summary.json").read_text()).get("demo", {}).get("title") or path.name
    except (OSError, ValueError):
        return path.name


def esc(s) -> str:
    return html.escape(str(s))


# ----------------------------------------------------------------- state -----
runs = (sorted(p.parent for p in (ROOT / "demo").glob("*/summary.json"))
        + sorted(p.parent for p in (ROOT / "outputs").glob("*/summary.json")))
if not runs:
    st.error("No runs found. Build one with `python -m oceansound.pipeline synthetic --out outputs/synthetic`.")
    st.stop()

qp = st.query_params
run_names = [p.name for p in runs]
if "run" not in st.session_state:
    st.session_state.run = qp.get("data") if qp.get("data") in run_names else run_names[0]
if "sheet" not in st.session_state:
    st.session_state.sheet = qp.get("sheet") if qp.get("sheet") in SHEETS else "S-1"

# ------------------------------------------------------------- sidebar -----
with st.sidebar:
    st.markdown('<div class="ds-block"><div class="ds-project">Ocean Soundscape'
                '<span>One year of deep-ocean sound from the MARS observatory, 890 m below '
                'Monterey Bay, California.</span></div></div>', unsafe_allow_html=True)
    SHORT = {"1_year_2018-19": "Year · Jul 2018 to Jun 2019", "2_week_2018-11": "Week · 1-7 Nov 2018",
             "synthetic": "Synthetic test bench", "synthetic_relative": "Synthetic, relative features"}
    st.selectbox("Data set", run_names, key="run", format_func=lambda n: SHORT.get(n, n))
    st.radio("Drawing index", list(SHEETS), key="sheet",
             format_func=lambda s: f"{s}   {SHEETS[s][0]}")

run = runs[run_names.index(st.session_state.run)]
df, ltsa, thumbs, freqs, summary, summ, conf = load_run(str(run))
demo = summary.get("demo", {})
named = demo.get("cluster_names", {})
clusters = sorted(int(c) for c in df["cluster"].unique())
name_of = {k: named.get(str(k), [f"Cluster {k}"])[0] for k in clusters}
evidence_of = {k: named.get(str(k), [None, ""])[1] for k in clusters}
color_of = {k: (T.BACKGROUND if "background" in name_of[k].lower() else T.pencil(i))
            for i, k in enumerate(clusters)}
units = summary.get("units", "dB re FS")
calibrated = "µPa" in units
seg_s = summary["seg_seconds"]
df["when"] = df["start_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
ids = df["segment_id"].tolist()
row_of = {s: i for i, s in enumerate(ids)}


def notable_minutes() -> list[tuple[str, str]]:
    """One-click examples: (label, segment_id)."""
    out = []
    if "mbari_blue_ci" in df and df["mbari_blue_ci"].notna().any():
        out.append(("Blue whale call", df.loc[df["mbari_blue_ci"].idxmax(), "segment_id"]))
    else:
        out.append(("Blue whale band", df.loc[df["blue_excess"].idxmax(), "segment_id"]))
    out.append(("Fin whale chorus", df.loc[df["fin_excess"].idxmax(), "segment_id"]))
    hump = [k for k, n in name_of.items() if "humpback" in n.lower()]
    pool = df[df["cluster"] == hump[0]] if hump else df
    out.append(("Humpback song" if hump else "Pulsed calls",
                pool.loc[pool["var_200_2k"].idxmax(), "segment_id"]))
    i_loud = df["b_10_100"].idxmax()
    loud, k_loud = df.loc[i_loud, "segment_id"], int(df.loc[i_loud, "cluster"])
    if loud == EARTHQUAKE_ID:
        text = "M3.3 earthquake"
    elif "transient" in name_of[k_loud].lower():
        text = "Loudest transient"
    else:
        text = "Loudest low tone"
    out.append((text, loud))
    return out


NOTABLE = notable_minutes()
if st.session_state.get("minute_run") != run.name or st.session_state.get("minute") not in row_of:
    wanted = qp.get("minute")
    st.session_state.minute = wanted if wanted in row_of else NOTABLE[0][1]
    st.session_state.minute_run = run.name


def select_minute(sid: str):
    st.session_state.minute = sid


def take_selection(event, key: str):
    """Apply a chart click once (charts keep their selection across reruns)."""
    try:
        pts = event["selection"]["points"]
    except (TypeError, KeyError):
        return
    if not pts:
        return
    p = pts[0]
    sid = p.get("customdata")
    if isinstance(sid, list):
        sid = sid[0]
    sig = (key, sid)
    if sid in row_of and st.session_state.get(f"last_{key}") != sig:
        st.session_state[f"last_{key}"] = sig
        st.session_state.minute = sid
        st.rerun()


sheet = st.session_state.sheet
st.query_params.update({"data": run.name, "sheet": sheet, "minute": st.session_state.minute})

with st.sidebar:
    n, k = summary["n_windows"], summary["k"]
    rows = [
        ("Drawn by", '<b>Paul Lai</b><br>M.S. Data Science<br>UC San Diego'),
        ("Contact", '<a href="mailto:holai@ucsd.edu">holai@ucsd.edu</a>'),
        ("Data", f'<span class="mono">{n:,} × {seg_s:g} s</span><br>'
                 f'{esc(summary["time_range"][0][:10])} to {esc(summary["time_range"][1][:10])}'),
        ("Model", f'<span class="mono">k = {k} · silhouette {summary["silhouette_by_k"][str(k)]:.2f}</span>'),
        ("Units", esc(units)),
        ("Checked", "NOAA buoy 46042<br>USGS earthquake catalog<br>MBARI call indices"),
        ("Revision", '<span class="mono">C</span>: fin whale background fixed'),
        ("Source", f'<a href="{REPO_URL}">GitHub</a>&emsp;<a href="{REPORT_URL}">Report</a>'),
    ]
    st.markdown('<div class="ds-block">' + "".join(
        f'<div class="row"><div class="k">{a}</div><div class="v">{b}</div></div>' for a, b in rows)
        + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------- pieces -----
def go_to(target: str, sid: str | None = None, data: str | None = None):
    """Button callback: open a sheet, optionally at a minute of a given data set."""
    if data and data in run_names and data != st.session_state.run:
        st.session_state.run = data
        st.session_state.minute_run = data
    if sid:
        st.session_state.minute = sid
    st.session_state.sheet = target


def go_sheet():
    if st.session_state.get("sheet_m"):
        st.session_state.sheet = st.session_state.sheet_m


def sheet_header(title: str, lede: str):
    st.markdown('<div class="ds-zones">' + "".join(f"<span>{n}</span>" for n in range(1, 9)) + "</div>",
                unsafe_allow_html=True)
    with st.container(key="mobile_nav"):
        st.session_state.sheet_m = sheet
        st.segmented_control("Sheet", list(SHEETS), key="sheet_m", on_change=go_sheet,
                             label_visibility="collapsed")
    st.markdown(f'<div class="ds-sheet"><div><h1>{esc(title)}</h1><p>{esc(lede)}</p></div>'
                f'<div class="ds-sheetno"><div class="k">SHEET</div><div class="n">{sheet}</div></div></div>',
                unsafe_allow_html=True)


def label(text: str, caption: str | None = None):
    st.markdown(f'<div class="ds-label">{esc(text)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="ds-caption">{caption}</div>', unsafe_allow_html=True)


def time_ticks():
    t = df["start_time"]
    if (t.max() - t.min()).days > 40:
        key = t.dt.strftime("%Y-%m")
        first = key.ne(key.shift()).to_numpy().nonzero()[0]
        first = [i for i in first if t.iloc[i].month in (1, 4, 7, 10)]      # quarterly: legible on phones
        return first, [t.iloc[i].strftime("%b") for i in first]      # the years go in the axis title
    key = t.dt.strftime("%Y-%m-%d")
    first = key.ne(key.shift()).to_numpy().nonzero()[0]
    return first, [str(t.iloc[i].day) for i in first]


def time_title() -> str:
    t = df["start_time"]
    if (t.max() - t.min()).days > 40:
        return f"{t.min():%B %Y} to {t.max():%B %Y}"
    return f"Day of {t.iloc[0]:%B %Y} (UTC)"


def elevation(height=330) -> go.Figure:
    """The long-term spectral average, dimensioned with the bands that matter."""
    lo, hi = np.percentile(ltsa, [30, 99.7])        # quiet stays bare paper; only loud takes ink
    x = np.arange(len(df))
    fig = go.Figure(go.Heatmap(
        z=ltsa.T, x=x, y=freqs, colorscale=T.DIAZO_SCALE, zmin=lo, zmax=hi, showscale=False,
        customdata=np.repeat(df["when"].to_numpy()[None, :], len(freqs), axis=0),
        hovertemplate="%{customdata}<br>%{y:.0f} Hz · %{z:.0f} dB<extra></extra>"))
    tv, tt = time_ticks()
    fig.update_yaxes(type="log", title="Hz", tickvals=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                     ticktext=["10", "20", "50", "100", "200", "500", "1k", "2k", "5k"])
    fig.update_xaxes(tickvals=tv, ticktext=tt, range=[-0.5, len(df) - 0.5], showgrid=False, title=time_title())
    T.style_figure(fig, height, margin=dict(l=64, r=62, t=8, b=48))
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(showgrid=False)
    if calibrated or "MARS" in str(df["source"].iloc[0]):
        T.dimension(fig, 15, 25, "fin<br>~20 Hz")
        T.dimension(fig, 40, 50, "blue<br>~45 Hz")
        T.dimension(fig, 1000, 7900, "wind<br>&gt;1 kHz", color=T.GRAPHITE_SOFT)
    i = row_of[st.session_state.minute]
    fig.add_vline(x=i, line=dict(color=T.RED, width=1.5))
    fig.add_annotation(x=i, y=1.0, yref="paper", text="A", showarrow=False, yanchor="bottom",
                       font=dict(family="Overpass", size=12, color=T.RED), xanchor="center")
    return fig


def lanes(height=None) -> go.Figure:
    """One lane per cluster, one mark per minute: click a mark to open it as Detail A."""
    fig = go.Figure()
    for k in clusters:
        d = df[df["cluster"] == k]
        fig.add_trace(go.Scatter(
            x=d.index, y=[name_of[k]] * len(d), mode="markers", name=name_of[k],
            marker=dict(symbol="line-ns", size=11, line=dict(width=1.6, color=color_of[k])),
            customdata=d["segment_id"], text=d["when"],
            hovertemplate="%{text}<br>" + esc(name_of[k]) + "<extra></extra>"))
    i = row_of[st.session_state.minute]
    fig.add_vline(x=i, line=dict(color=T.RED, width=1.5))
    tv, tt = time_ticks()
    fig.update_xaxes(tickvals=tv, ticktext=tt, range=[-0.5, len(df) - 0.5], showgrid=False)
    fig.update_yaxes(categoryorder="array", categoryarray=[name_of[k] for k in reversed(clusters)],
                     showgrid=True, griddash="solid", gridcolor=T.INK_FAINT, showticklabels=False)
    T.style_figure(fig, height or 50 * len(clusters) + 56,
                   margin=dict(l=64, r=62, t=4, b=48))
    fig.update_xaxes(tickangle=0, title_text=time_title())
    for k in clusters:                       # lane names written into the lane, left edge
        fig.add_annotation(xref="paper", yref="y", x=0.006, y=name_of[k], text=esc(name_of[k]),
                           showarrow=False, xanchor="left", yanchor="bottom", yshift=9,
                           bgcolor="rgba(236,238,233,0.86)", borderpad=1,
                           font=dict(family="Overpass", size=10.5, color=T.GRAPHITE))
    fig.update_layout(clickmode="event+select")
    return fig


def spectrogram(i: int, height=300) -> go.Figure:
    S = thumbs[i]
    t = np.linspace(0, seg_s, S.shape[1])
    fig = go.Figure(go.Heatmap(z=S, x=t, y=freqs, colorscale=T.DIAZO_SCALE, showscale=False,
                               zmin=np.percentile(thumbs, 2), zmax=np.percentile(thumbs, 99.7),
                               hovertemplate="%{x:.0f} s · %{y:.0f} Hz · %{z:.0f} dB<extra></extra>"))
    fig.update_yaxes(type="log", title="Hz", tickvals=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                     ticktext=["10", "20", "50", "100", "200", "500", "1k", "2k", "5k"], showgrid=False)
    fig.update_xaxes(title="Seconds", showgrid=False)
    T.style_figure(fig, height, margin=dict(l=48, r=52, t=6, b=40))
    if calibrated or "MARS" in str(df["source"].iloc[0]):
        T.dimension(fig, 15, 25, "fin")
        T.dimension(fig, 40, 50, "blue")
    return fig


def chip(k: int) -> str:
    return f'<span class="ds-chip" style="color:{color_of[k]}">{esc(name_of[k])}</span>'


def listen_controls(i: int, key: str, compact=False):
    sid = ids[i]
    source = str(df.loc[i, "source"])
    if source == "synthetic":
        st.caption("Synthetic test minutes are generated in memory and not stored as audio.")
        return
    c1, c2 = st.columns([1.1, 1]) if not compact else (st.container(), st.container())
    with c2:
        speed = st.segmented_control("Playback speed", [1, 4, 10], default=10, key=f"speed_{key}",
                                     format_func=lambda v: f"{v}×", label_visibility="collapsed",
                                     help="At 10× speed a 20 Hz fin whale pulse becomes 200 Hz, "
                                          "audible on laptop speakers.") or 10
    with c1:
        play = st.button(f"Listen at {speed}×", key=f"play_{key}_{sid}", type="primary", width="stretch")
    if play:
        try:
            x, fs = load_audio(source, pd.Timestamp(df.loc[i, "start_time"]), seg_s)
            y, play_fs = for_listening(x, fs, speed)
            st.audio(y, sample_rate=play_fs, autoplay=True)
            st.caption(f"{len(x) / fs:.0f} s recorded at {fs / 1000:g} kHz, played in {len(x) / fs / speed:.0f} s. "
                       "Streamed from MBARI's archive and peak-normalized.")
        except Exception as e:  # noqa: BLE001 - the archive can be unreachable; say so plainly
            st.error(f"The MBARI archive did not answer ({type(e).__name__}). Try again in a moment.")


def detail_card(compact=True):
    i = row_of[st.session_state.minute]
    k = int(df.loc[i, "cluster"])
    st.markdown(f'<div class="ds-detail-tag">{T.detail_bubble("A", "S-3")}<div>'
                f'<div class="t">{esc(df.loc[i, "when"])}</div>'
                f'<div class="s">{chip(k)}</div></div></div>', unsafe_allow_html=True)
    st.plotly_chart(spectrogram(i, 250 if compact else 380), config=PLOT_CONFIG, width="stretch",
                    key=f"spec_{'c' if compact else 'f'}")
    listen_controls(i, "c" if compact else "f", compact=compact)


def notable_buttons(key: str):
    cols = st.columns(len(NOTABLE))
    for c, (text, sid) in zip(cols, NOTABLE):
        c.button(text, key=f"jump_{key}_{text}", on_click=select_minute, args=(sid,), width="stretch",
                 type="primary" if st.session_state.minute == sid else "secondary")


def schedule_html(rows: list[list[str]], head: list[str], numeric: set[int], hi_row: int | None = None,
                  cls="ds-schedule") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = ""
    for r, cells in enumerate(rows):
        tds = "".join(f'<td class="n">{c}</td>' if j in numeric else f"<td>{c}</td>" for j, c in enumerate(cells))
        body += f'<tr class="{"hi" if r == hi_row else ""}">{tds}</tr>'
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------- guide -----
TOUR = ["Listen", "Elevation", "Spectrogram", "Clusters", "Evidence", "Explore"]

GLOSSARY = [
    ("Hydrophone", "An underwater microphone. This one sits on the MARS cabled observatory, 890 m deep in "
                   "Monterey Bay, and records 16,000 samples per second."),
    ("dB re 1 µPa", "Sound pressure level in decibels relative to one micropascal, the standard reference in "
                    "underwater acoustics. +10 dB is ten times the power."),
    ("Spectrogram", "Frequency against time for one recording: here one minute, with darker ink where a "
                    "frequency is louder."),
    ("LTSA", "Long-term spectral average: one averaged spectrum per minute, laid side by side over days or "
             "months. The elevation on S-1."),
    ("Band level", "Mean level inside a frequency band, for example 1-8 kHz, where wind-driven surface noise lives."),
    ("Excess", "Level in a call band minus the level of its neighbours. A whale call is narrow and sticks out; "
               "wind and ships are broad and cancel."),
    ("Kurtosis", "How spiky the waveform is. Near 0 for steady noise; large for pulses and earthquakes."),
    ("PCA", "Principal component analysis: rotates the 16 features into a few directions that keep most of the "
            "variation. S-2 plots the first two."),
    ("K-Means", "Groups minutes so each is close to its group's centre. k, the number of groups, is chosen by "
                "the silhouette score."),
    ("Silhouette", "From -1 to 1: how much closer a minute is to its own cluster than to the next one."),
    ("Call index", "MBARI's whale call measure: level at the call frequency divided by the level beside it. "
                   "1 means no call. Used here only to check the clusters, never to build them."),
    ("η² (eta squared)", "Share of a variable's variance explained by the clusters. Used to show that the "
                         "clusters stopped tracking wind."),
]

FAQ = [
    ("Why can't I hear anything at 1×?", "Fin and blue whale calls sit at 15-50 Hz, below what laptop speakers "
     "reproduce. At 10× a 20 Hz pulse plays at 200 Hz and the minute lasts 6 s."),
    ("Why one minute per hour, every 4th day?", "A full year is about 1.4 TB of audio. Sampling 2,189 minutes "
     "keeps the seasons and the daily cycle while streaming only a few GB."),
    ("Are the cluster names certain?", "Fin and blue are checked against MBARI's call indices. Humpback is "
     "named from spectrograms and listening only, and says so."),
    ("Is this real data?", "Yes. Every sound is streamed on demand from MBARI's public archive on AWS Open Data; "
     "the spectrogram thumbnails here are compressed previews of it."),
    ("Can I run it myself?", "Yes. The code, tests and reports are on GitHub; the commands are below."),
]


def tour_move(step: int):
    st.session_state.tour_step = TOUR[max(0, min(len(TOUR) - 1, step))]


def quiz(key: str, question: str, options: list[str], answer: int, why: str):
    st.markdown(f'<div class="ds-label" style="margin-top:1rem">Check yourself</div>'
                f'<div class="ds-quiz-q">{esc(question)}</div>', unsafe_allow_html=True)
    pick = st.radio(question, options, index=None, key=f"quiz_{key}", label_visibility="collapsed")
    if pick is not None:
        ok = options.index(pick) == answer
        st.markdown(f'<div class="ds-quiz-a {"ok" if ok else "no"}"><b>{"Right." if ok else "Not quite."}</b> '
                    f'{esc(why)}</div>', unsafe_allow_html=True)


def render_guide():
    sheet_header("How to read this drawing set",
                 "Six short steps, each with the real data beside it. About two minutes; every step "
                 "ends with a button that opens the full sheet.")
    year = run.name == "1_year_2018-19"
    st.session_state.setdefault("tour_step", TOUR[0])
    st.segmented_control("Tour step", TOUR, key="tour_step", label_visibility="collapsed",
                         format_func=lambda s: f"{TOUR.index(s) + 1}  {s}")
    step = TOUR.index(st.session_state.tour_step or TOUR[0])
    blue_sid, fin_sid = NOTABLE[0][1], NOTABLE[1][1]

    text, fig = st.columns([1, 1.15], gap="large")
    with text:
        st.markdown(f'<div class="ds-tourno"><span>{step + 1}</span><small>of {len(TOUR)}</small></div>',
                    unsafe_allow_html=True)
        if step == 0:
            st.markdown("""<div class="ds-guide"><h3>Listen first</h3>
<p>Everything here starts as sound: one minute recorded by a hydrophone 890 m below Monterey Bay.
The spectrogram on this sheet is the minute with the strongest blue whale call of the data set.</p>
<p>Press <b>Listen</b> at 1×, then at 10×. Blue and fin whales call between 15 and 50 Hz, below what most
speakers can play. Playing ten times faster lifts a 20 Hz call to 200 Hz and squeezes the minute into 6 s.</p></div>""",
                        unsafe_allow_html=True)
            quiz("listen", "Why play the minute at 10×?",
                 ["To make it shorter", "To lift the calls into the range speakers can play", "To filter out noise"],
                 1, "Speed multiplies every frequency: 20 Hz becomes 200 Hz. Shorter is a side effect.")
            st.button("Open S-1 at this minute", key="g_go0", on_click=go_to, args=("S-1", blue_sid))
        elif step == 1:
            st.markdown(f"""<div class="ds-guide"><h3>Read the elevation</h3>
<p>The elevation is a long-term spectral average: <b>one thin column per sampled minute</b>, time running left
to right, frequency rising on a log scale from 10 Hz to 7.9 kHz. Darker ink is louder.</p>
<ol class="ds-notes">
<li><span><b>~20 Hz</b>: fin whale chorus, a dark band {"from September, strongest October-November" if year else "through the week"}.</span></li>
<li><span><b>~45 Hz</b>: blue whale B calls.</span></li>
<li><span><b>Above 1 kHz</b>: wind on the sea surface. Vertical stripes are windy days.</span></li>
</ol></div>""", unsafe_allow_html=True)
            quiz("ltsa", "Where should a fin whale chorus appear?",
                 ["A dark band near 20 Hz", "Vertical stripes above 1 kHz", "A single dot at 7.9 kHz"],
                 0, "Fin whales repeat 20 Hz pulses for hours, so the band stays dark over many columns. "
                    "Stripes above 1 kHz follow the wind.")
            st.button("Open S-1", key="g_go1", on_click=go_to, args=("S-1",))
        elif step == 2:
            st.markdown("""<div class="ds-guide"><h3>Read one minute</h3>
<p>A spectrogram zooms into a single minute: time across, frequency up, ink for level. The spectrogram on this sheet is the
minute with the strongest fin whale signal.</p>
<ol class="ds-notes">
<li><span><b>Fin whale</b>: short pulses near 20 Hz, repeated through the minute.</span></li>
<li><span><b>Blue whale</b>: a long, steady tone just below 50 Hz, with fainter harmonics above it.</span></li>
<li><span><b>Ship or wind</b>: broad energy across many frequencies at once, not a narrow line.</span></li>
</ol></div>""", unsafe_allow_html=True)
            quiz("spec", "A minute shows energy spread evenly from 100 Hz to 5 kHz. Most likely?",
                 ["A whale call", "Wind or a passing ship", "A recording gap"],
                 1, "Calls are narrow lines. Broadband energy is weather or ships; a gap would be blank.")
            st.button("Open this minute on S-3", key="g_go2", on_click=go_to, args=("S-3", fin_sid))
        elif step == 3:
            st.markdown("""<div class="ds-guide"><h3>How the minutes were grouped</h3>
<p>Each minute becomes 16 numbers: band levels, how far each call band stands out from its neighbours,
how much it fluctuates, spectral shape and spikiness. K-Means then groups similar minutes, with no labels
given.</p>
<p>On all 16 features the groups simply follow the weather: calm days against windy days. Using only the six
<b>relative</b> features, which compare each band with its neighbours, cancels the wind (η² from
<span class="ds-val">0.49</span> to <span class="ds-val">0.020</span>) and lets the animals separate.</p></div>""",
                        unsafe_allow_html=True)
            quiz("cluster", "Why cluster on relative features?",
                 ["They are faster to compute", "Wind raises all bands together, so differences cancel it",
                  "MBARI requires it"],
                 1, "Wind lifts the whole spectrum. A call lifts one narrow band above its neighbours, "
                    "which a difference keeps.")
            st.button("Open S-2", key="g_go3", on_click=go_to, args=("S-2",))
        elif step == 4:
            st.markdown("""<div class="ds-guide"><h3>Check every claim</h3>
<p>Clusters only describe; each interpretation was tested against data the model never saw.</p>
<ol class="ds-notes">
<li><span><b>Wind</b>: 1-8 kHz level follows NOAA buoy 46042 (<span class="ds-val">r = 0.86</span> over the year).</span></li>
<li><span><b>Earthquake</b>: the loudest minute of 1-7 Nov 2018 matches an M3.3 quake in the USGS catalog.</span></li>
<li><span><b>Whales</b>: MBARI's own call indices agree month by month (<span class="ds-val">r = 0.99</span> fin,
<span class="ds-val">0.91</span> blue).</span></li>
<li><span><b>A flaw, fixed</b>: that comparison exposed blue whale harmonics in my fin whale feature; moving
the background bands raised agreement from <span class="ds-val">0.81</span> to <span class="ds-val">0.93</span>.</span></li>
</ol></div>""", unsafe_allow_html=True)
            st.button("Open S-4", key="g_go4", on_click=go_to, args=("S-4",))
        else:
            st.markdown("""<div class="ds-guide"><h3>Explore on your own</h3>
<ol class="ds-notes">
<li><span><b>Click any mark</b> in the cluster lanes (S-1) or any point in the PCA plan (S-2) to open that minute.</span></li>
<li><span><b>Step through a cluster</b> with Previous and Next on S-3, and compare its features with the cluster average.</span></li>
<li><span><b>Switch data set</b> in the sidebar: the year shows seasons, the week shows single events.</span></li>
<li><span><b>Share a view</b>: the address bar keeps the data set, sheet and minute, so a link opens exactly what you see.</span></li>
</ol></div>""", unsafe_allow_html=True)
        st.write("")
        b1, b2 = st.columns(2)
        b1.button("Back", key="g_prev", on_click=tour_move, args=(step - 1,), disabled=step == 0,
                  width="stretch")
        b2.button("Next step", key="g_next", on_click=tour_move, args=(step + 1,), disabled=step == len(TOUR) - 1,
                  width="stretch", type="primary")

    with fig:
        if step == 0:
            i = row_of[blue_sid]
            label(f"{NOTABLE[0][0]} · {df.loc[i, 'when']}")
            st.plotly_chart(spectrogram(i, 300), config=PLOT_CONFIG, width="stretch", key="g_spec0")
            listen_controls(i, "g0")
        elif step == 1:
            label("Elevation", f"Levels in {esc(units)}. The red line marks the selected minute.")
            st.plotly_chart(elevation(360), config=PLOT_CONFIG, width="stretch", key="g_elev")
        elif step == 2:
            i = row_of[fin_sid]
            label(f"{NOTABLE[1][0]} · {df.loc[i, 'when']}")
            st.plotly_chart(spectrogram(i, 320), config=PLOT_CONFIG, width="stretch", key="g_spec2")
            listen_controls(i, "g2")
        elif step == 3:
            label("Cluster lanes", "Each mark is one minute, placed in its cluster's lane.")
            st.plotly_chart(lanes(), config=PLOT_CONFIG, width="stretch", key="g_lanes")
            shares = df["cluster"].value_counts(normalize=True)
            st.markdown(" ".join(f'{chip(k)} <span class="ds-val">{shares[k]:.0%}</span>&emsp;' for k in clusters),
                        unsafe_allow_html=True)
        elif step == 4:
            label("Plate 1. Agreement with MBARI's call indices")
            st.image(str(ROOT / "docs" / "img" / "year_mbari_ci_comparison.png"), width="stretch")
        else:
            label("Try these")
            tries = [("Hear the M3.3 earthquake", "S-3", EARTHQUAKE_ID, "2_week_2018-11"),
                     ("Strongest blue whale call", "S-3", blue_sid, None),
                     ("Strongest fin whale chorus", "S-3", fin_sid, None),
                     ("Browse the clusters", "S-2", None, None)]
            for n_, (txt, target, sid, data) in enumerate(tries):
                if data and data not in run_names:
                    continue
                st.button(txt, key=f"g_try{n_}", on_click=go_to, args=(target, sid, data), width="stretch")

    st.write("")
    st.markdown('<div class="ds-sheet ds-sub"><div><h1>Handbook</h1>'
                '<p>Terms, sheets and common questions.</p></div></div>', unsafe_allow_html=True)
    h1, h2 = st.columns([1.2, 1], gap="large")
    with h1:
        label("Terms")
        st.markdown(schedule_html([[f"<b>{esc(t)}</b>", esc(d)] for t, d in GLOSSARY], ["Term", "Meaning"], set()),
                    unsafe_allow_html=True)
    with h2:
        label("The sheets")
        st.markdown(schedule_html([[f'<span class="mono">{s_}</span>', esc(SHEETS[s_][0]), esc(SHEETS[s_][1])]
                                   for s_ in SHEETS if s_ != "G-0"], ["Sheet", "Title", "Use it to"], set()),
                    unsafe_allow_html=True)
        label("Questions")
        st.markdown('<ol class="ds-notes">' + "".join(
            f"<li><span><b>{esc(q)}</b> {esc(a)}</span></li>" for q, a in FAQ) + "</ol>", unsafe_allow_html=True)
        label("Run it yourself")
        st.code("git clone https://github.com/Paul-Lai0111/ocean-soundscape-ml\n"
                "cd ocean-soundscape-ml && pip install -e \".[dev]\"\n"
                "pytest -q                                   # offline\n"
                "python -m oceansound.pipeline mbari --dates 2018-11-01 --out outputs/day\n"
                "streamlit run app/dashboard.py", language="bash")


# --------------------------------------------------------------- sheets -----
title, lede = SHEETS[sheet]
HEADLINES = {"1_year_2018-19": "A year beneath Monterey Bay", "2_week_2018-11": "One week in November 2018"}
LEDES = {
    "1_year_2018-19": "One minute per hour, every 4th day, July 2018 to June 2019. Whale song fills about half "
                      "of all minutes in October-November and almost none from March to August.",
    "2_week_2018-11": "One minute every 30 minutes, 1-7 November 2018. Four clusters: background, humpback "
                      "song, blue whale calls and impulsive events, including an M3.3 earthquake.",
}
if sheet == "G-0":
    render_guide()

elif sheet == "S-1":
    sheet_header(HEADLINES.get(run.name, demo.get("title", run.name) if demo else run.name),
                 LEDES.get(run.name, demo.get("description", lede) if demo else lede))
    s1 = st.container(key="s1_main")
    left, right = s1.columns([2.15, 1], gap="large")
    with left:
        label("Elevation: long-term spectral average",
              f"One column per sampled minute; ink density is level ({esc(units)}). "
              "The red line marks Detail A.")
        st.plotly_chart(elevation(), config=PLOT_CONFIG, width="stretch", key="elev")
        label("Cluster lanes", "Each mark is one minute. Click a mark to open it as Detail A.")
        ev = st.plotly_chart(lanes(), config=PLOT_CONFIG, width="stretch", key="lanes_s1",
                             on_select="rerun", selection_mode="points")
        take_selection(ev, "lanes_s1")
    with right:
        label("Detail A: one minute")
        detail_card()
        label("Jump to")
        for text, sid in NOTABLE:
            st.button(text, key=f"jump_s1_{text}", on_click=select_minute, args=(sid,), width="stretch",
                      type="primary" if st.session_state.minute == sid else "secondary")
        st.button("New here? Six-step tour on G-0", key="to_guide", type="tertiary",
                  on_click=go_to, args=("G-0",))

    st.write("")
    n1, n2 = st.columns([1.25, 1], gap="large")
    with n1:
        label("General notes")
        st.markdown("""<ol class="ds-notes">
<li><span><b>Whale song has a season.</b> Fin, blue and humpback indicators rise from September,
peak in October-November and fade by January; MBARI's own call indices agree month by month
(<span class="ds-val">r = 0.99</span> fin, <span class="ds-val">0.91</span> blue).</span></li>
<li><span><b>Wind dominates the raw sound.</b> Noise above 1 kHz follows buoy wind speed
(<span class="ds-val">r = 0.86</span>), so clustering on all features finds calm and windy days.</span></li>
<li><span><b>Relative features remove the weather.</b> Comparing each band with its neighbours cuts the wind
dependence of the clusters from <span class="ds-val">η² 0.49</span> to <span class="ds-val">0.020</span>.</span></li>
<li><span><b>An outlier was an earthquake.</b> The loudest minute of 1-7 Nov 2018 is an M3.3 quake 78 km away,
arriving <span class="ds-val">~17 s</span> after origin time.</span></li>
<li><span><b>A flaw in my own feature was found and fixed.</b> Blue whale harmonics sat inside the fin whale
background bands; the fix raised agreement with MBARI from <span class="ds-val">0.81</span> to
<span class="ds-val">0.93</span>. REV_C_MARK</span></li>
</ol>""".replace("REV_C_MARK", T.delta("C")), unsafe_allow_html=True)
    with n2:
        label("Cluster schedule")
        shares = df["cluster"].value_counts(normalize=True)
        rows = [[f'<span class="ds-swatch" style="background:{color_of[k]}"></span><b>{esc(name_of[k])}</b>'
                 f'<br><span style="color:var(--graphite-soft);font-size:.78rem">{esc(evidence_of[k])}</span>',
                 f"{(df['cluster'] == k).sum():,}", f"{shares[k]:.0%}"] for k in clusters]
        st.markdown(schedule_html(rows, ["Cluster", "Minutes", "Share"], {1, 2}), unsafe_allow_html=True)

elif sheet == "S-2":
    sheet_header(title, lede)
    a, b = st.columns([1.35, 1], gap="large")
    with a:
        label("Plan: minutes in PCA space", "Click a point to open it as Detail A on sheet S-3.")
        fig = go.Figure()
        for k in clusters:
            d = df[df["cluster"] == k]
            quiet = "background" in name_of[k].lower()
            fig.add_trace(go.Scatter(x=d["PC1"], y=d["PC2"], mode="markers", name=name_of[k],
                                     marker=dict(size=4 if quiet else 6, color=color_of[k],
                                                 opacity=0.35 if quiet else 0.85, line=dict(width=0)),
                                       customdata=d["segment_id"], text=d["when"],
                                       hovertemplate="%{text}<br>" + esc(name_of[k]) + "<extra></extra>"))
        i = row_of[st.session_state.minute]
        fig.add_trace(go.Scatter(x=[df.loc[i, "PC1"]], y=[df.loc[i, "PC2"]], mode="markers+text",
                                 text=["A"], textposition="top right", hoverinfo="skip", showlegend=False,
                                 textfont=dict(color=T.RED, family="Overpass", size=13),
                                 marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=T.RED, width=1.5))))
        ev_ = summary.get("pca_explained_variance", [0, 0])
        fig.update_xaxes(title=f"PC1 · {ev_[0]:.0%} of variance")
        fig.update_yaxes(title=f"PC2 · {ev_[1]:.0%} of variance")
        for k in clusters:                   # direct labels at each cluster's centre instead of a legend
            d = df[df["cluster"] == k]
            fig.add_annotation(x=d["PC1"].median(), y=d["PC2"].median(), text=f"<b>{esc(name_of[k])}</b>",
                               showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=color_of[k], ax=34, ay=-26,
                               bgcolor="rgba(236,238,233,0.9)", borderpad=2,
                               font=dict(family="Overpass", size=11, color=color_of[k]))
        T.style_figure(fig, 460, margin=dict(l=56, r=16, t=8, b=44))
        ev = st.plotly_chart(fig, config=PLOT_CONFIG, width="stretch", key="plan",
                             on_select="rerun", selection_mode="points")
        take_selection(ev, "plan")
    with b:
        label("Model selection", "Silhouette for each number of clusters; the chosen k is marked.")
        sil = pd.Series({int(kk): v for kk, v in summary["silhouette_by_k"].items()}).sort_index()
        f2 = go.Figure(go.Scatter(x=sil.index, y=sil.values, mode="lines+markers",
                                  line=dict(color=T.INK, width=1.5), marker=dict(size=7, color=T.INK)))
        f2.add_vline(x=summary["k"], line=dict(color=T.RED, width=1.5, dash="dot"))
        f2.update_xaxes(title="k", dtick=1)
        f2.update_yaxes(title="Silhouette")
        st.plotly_chart(T.style_figure(f2, 200), config=PLOT_CONFIG, width="stretch", key="sil")
        if conf is not None:
            label("Synthetic truth vs cluster")
            st.markdown(schedule_html([[esc(r)] + [str(v) for v in conf.loc[r]] for r in conf.index],
                                      ["True source"] + [f"C{c}" for c in conf.columns],
                                      set(range(1, len(conf.columns) + 1))), unsafe_allow_html=True)
        label("When each cluster occurs")
        t = df["start_time"]
        per = t.dt.strftime("%Y-%m") if (t.max() - t.min()).days > 40 else t.dt.strftime("%d %b")
        share = pd.crosstab(per, df["cluster"], normalize="index")
        f3 = go.Figure()
        for kk in clusters:
            f3.add_trace(go.Bar(x=share.index, y=share[kk], name=name_of[kk], marker_color=color_of[kk],
                                marker_line_width=0, hovertemplate="%{x}: %{y:.0%}<extra>" + esc(name_of[kk]) + "</extra>"))
        f3.update_layout(barmode="stack", bargap=0.15)
        f3.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(T.style_figure(f3, 230, margin=dict(l=48, r=10, t=8, b=36)),
                        config=PLOT_CONFIG, width="stretch", key="share")

    label("Cluster schedule: feature means",
          "Read each row as a fingerprint. The largest value in each column is marked in red pencil.")
    cols = [c for c in ["fin_excess", "blue_excess", "var_200_2k", "p90_minus_med_200_2k", "b_1k_8k",
                        "kurtosis", "mbari_fin_ci", "mbari_blue_ci"] if c in summ.columns and summ[c].notna().any()]
    heads = ["Cluster", "Minutes"] + [FEATURE_NAMES.get(c, c.replace("mbari_", "MBARI ").replace("_ci", " CI"))
                                      for c in cols]
    rows = []
    for kk in clusters:
        cells = [f'<span class="ds-swatch" style="background:{color_of[kk]}"></span><b>{esc(name_of[kk])}</b>',
                 f"{int(summ.loc[kk, 'n_windows']):,}"]
        for c in cols:
            v = summ.loc[kk, c]
            top = v == summ.loc[clusters, c].max()
            cells.append(f'<span style="color:{T.RED};font-weight:700">{v:.2f}</span>' if top else f"{v:.2f}")
        rows.append(cells)
    st.markdown(schedule_html(rows, heads, set(range(1, len(heads)))), unsafe_allow_html=True)

elif sheet == "S-3":
    sheet_header(title, lede)
    i = row_of[st.session_state.minute]
    nav1, nav2, nav3, nav4 = st.columns([1.4, 2.5, 0.85, 0.85], vertical_alignment="bottom")
    with nav1:
        pick = st.selectbox("Cluster", ["All clusters"] + [name_of[k] for k in clusters],
                            index=0, key="s3_cluster")
    pool = df if pick == "All clusters" else df[df["cluster"] == [k for k in clusters if name_of[k] == pick][0]]
    options = pool["segment_id"].tolist()
    if st.session_state.minute not in options:
        options = [st.session_state.minute] + options
    with nav2:
        st.selectbox("Minute", options, key="minute",
                     format_func=lambda s: f"{df.loc[row_of[s], 'when']}  ·  {name_of[int(df.loc[row_of[s], 'cluster'])]}")
    pos = options.index(st.session_state.minute)
    nav3.button("Previous", on_click=select_minute, args=(options[max(pos - 1, 0)],), width="stretch",
                disabled=pos == 0)
    nav4.button("Next", on_click=select_minute, args=(options[min(pos + 1, len(options) - 1)],),
                width="stretch", disabled=pos == len(options) - 1)
    notable_buttons("s3")
    st.write("")
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        label("Detail A: spectrogram", "Ink density is level; frequency is logarithmic.")
        detail_card(compact=False)
    with b:
        i = row_of[st.session_state.minute]
        k = int(df.loc[i, "cluster"])
        label("Bill of measurements",
              "This minute against its cluster and all minutes. Red pencil: furthest from the average.")
        feats = [f for f in summary["features"] if f in df.columns]
        mean_all, std_all = df[feats].mean(), df[feats].std().replace(0, 1)
        z = ((df.loc[i, feats].astype(float) - mean_all) / std_all).abs()
        top = set(z.sort_values(ascending=False).index[:3])
        rows = []
        for f in feats:
            v = float(df.loc[i, f])
            val = f'<span style="color:{T.RED};font-weight:700">{v:.2f}</span>' if f in top else f"{v:.2f}"
            rows.append([esc(FEATURE_NAMES.get(f, f)), val, f"{summ.loc[k, f]:.2f}", f"{mean_all[f]:.2f}"])
        st.markdown(schedule_html(rows, ["Feature", "This minute", "Cluster", "All"], {1, 2, 3}),
                    unsafe_allow_html=True)
        if pd.notna(df.loc[i, "label"]):
            st.caption(f"Synthetic source: {df.loc[i, 'label']}")
        ci = [c for c in ("mbari_fin_ci", "mbari_blue_ci") if c in df and pd.notna(df.loc[i, c])]
        if ci:
            st.markdown('<div class="ds-caption" style="margin-top:.6rem">MBARI call index (1 = no call): '
                        + " · ".join(f'{"fin" if "fin" in c else "blue"} <b class="mono">{df.loc[i, c]:.3f}</b>'
                                     for c in ci) + "</div>", unsafe_allow_html=True)

elif sheet == "S-4":
    sheet_header(title, lede)
    label("Revision table")
    revs = [
        ["A", "Recording gaps", "MBARI fills gaps with zeros; one minute with only 2% zeros passed silently. "
              "Minutes with more than 1% exact zeros are now skipped.", T.revision_cloud("gap detection")],
        ["B", "Calibration", "Levels converted to dB re 1 µPa from the hydrophone's lab calibration "
              "(+187.44 dB, ±0.55 dB).", T.revision_cloud("dB re 1 µPa")],
        ["C", "Fin whale background", "The comparison with MBARI exposed blue whale harmonics (~14 and ~28 Hz) "
              "inside the fin whale background bands. Moved to 11-13 and 33-36 Hz; agreement 0.81 to 0.93.",
         T.revision_cloud("feature v2")],
    ]
    st.markdown(schedule_html(revs, ["Rev", "Change", "Description", "Mark"], set(), cls="ds-schedule ds-rev")
                .replace("<td>A</td>", f'<td class="rev">{T.delta("A")}</td>')
                .replace("<td>B</td>", f'<td class="rev">{T.delta("B")}</td>')
                .replace("<td>C</td>", f'<td class="rev">{T.delta("C")}</td>'), unsafe_allow_html=True)
    st.write("")
    plates = [
        ("year_mbari_ci_comparison.png", "Plate 1. Agreement with MBARI's call indices",
         "Monthly r = 0.99 (fin) and 0.91 (blue) over 2,189 minutes. MBARI's index was re-implemented line by line "
         "and matches MBARI's own 2 kHz product to within 0.0006."),
        ("nov2018_wind_vs_noise.png", "Plate 2. Wind explains the default clusters",
         "1-8 kHz level against NOAA buoy 46042 wind, 1-7 Nov 2018: Spearman r = 0.84 (0.86 over the year)."),
        ("nov2018_earthquake.png", "Plate 3. An earthquake found as an outlier",
         "USGS: M3.3, 16:19:30 UTC, 12 km S of Tres Pinos, CA, ~78 km away. Energy arrives ~17 s after origin, "
         "between the expected P and S arrivals."),
        ("year_fin_index_investigation.png", "Plate 4. Revision C, located by ablation",
         "Index variants changing one design choice at a time: the background bands, not the peak width, "
         "caused the January-May gap."),
    ]
    for (f1, t1, c1), (f2, t2, c2) in zip(plates[::2], plates[1::2]):
        p1, p2 = st.columns(2, gap="large")
        for col, f, t_, c_ in ((p1, f1, t1, c1), (p2, f2, t2, c2)):
            with col:
                st.markdown(f'<div class="ds-label">{esc(t_)}</div>', unsafe_allow_html=True)
                st.image(str(ROOT / "docs" / "img" / f), width="stretch")
                st.markdown(f'<div class="ds-caption">{esc(c_)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ds-caption">Full method and every number: '
                f'<a href="{REPORT_URL}">the report</a> (English) and '
                f'<a href="{REPO_URL}/blob/main/docs/analysis_zh.md">中文版</a>.</div>', unsafe_allow_html=True)

elif sheet == "S-5":
    sheet_header(title, lede)
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        label("Bands")
        st.markdown(schedule_html([[f'<span class="mono">{esc(key)}</span>', f"{lo:g}-{hi:g} Hz", esc(d)]
                                   for key, (lo, hi, d) in BANDS.items()], ["Feature", "Band", "Targets"], set()),
                    unsafe_allow_html=True)
        label("Other features")
        st.markdown("""<ol class="ds-notes">
<li><span><b>Excess</b> (fin, blue): mean level in the call band minus its neighbouring bands, in dB.
Narrow-band calls stand out; broadband wind and ship noise do not.</span></li>
<li><span><b>Fluctuation</b>: how much band energy varies over one-second frames. Pulsed calls fluctuate;
steady noise does not.</span></li>
<li><span><b>Spectral shape</b>: centroid (power-weighted mean frequency) and flatness (0 dB is noise-like).</span></li>
<li><span><b>Kurtosis</b>: how impulsive the waveform is; 0 for Gaussian noise.</span></li>
</ol>""", unsafe_allow_html=True)
    with b:
        label("Method")
        st.markdown("""<ol class="ds-notes">
<li><span>Minutes are streamed from MBARI's public archive by byte range (~3 MB each), never whole 4 GB days.</span></li>
<li><span>Sixteen features per minute; the <b>relative</b> six are used for these clusters so that wind cancels out.</span></li>
<li><span>Standardize, clip to the 1st-99th percentile, PCA to 95% of variance, K-Means with k chosen by silhouette.</span></li>
<li><span>Checked against NOAA buoy wind, the USGS catalog, the hydrophone calibration and MBARI's call indices.</span></li>
<li><span>Not yet established: humpback is named from spectrograms only; one year, sampled every 4th day.</span></li>
</ol>""", unsafe_allow_html=True)
        label("Sources")
        st.markdown("""<div class="ds-caption">MBARI Pacific Sound (AWS Open Data), doi:10.1109/OCEANS.2016.7761363 ·
NOAA NDBC station 46042 · USGS Earthquake Catalog · MBARI Pacific Sound tutorials (J. Ryan).</div>""",
                    unsafe_allow_html=True)
        label("Author")
        st.markdown('<div class="ds-caption"><b style="color:var(--graphite)">Paul Lai</b>, M.S. Data Science, '
                    'UC San Diego · <a href="mailto:holai@ucsd.edu">holai@ucsd.edu</a> · '
                    f'<a href="{REPO_URL}">github.com/Paul-Lai0111</a></div>', unsafe_allow_html=True)

with st.container(key="mobile_foot"):
    st.markdown('<div class="ds-caption" style="margin-top:1.4rem;border-top:1px solid var(--ink);padding-top:.6rem">'
                '<b style="color:var(--graphite)">Drawn by Paul Lai</b>, M.S. Data Science, UC San Diego · '
                f'<a href="mailto:holai@ucsd.edu">holai@ucsd.edu</a> · <a href="{REPO_URL}">GitHub</a></div>',
                unsafe_allow_html=True)
