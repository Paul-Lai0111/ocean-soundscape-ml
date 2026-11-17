"""Visual system for the dashboard: an engineering drawing set printed on diazo whiteprint paper.

Paper is the ground, violet-blue diazo line is the ink, graphite carries text, and red pencil marks
revisions and the one thing the reader should look at. Charts follow the same rules: hairline grids,
outside tick marks, dimension lines instead of legends where possible.
"""

from __future__ import annotations

import math

import plotly.graph_objects as go

PAPER = "#eceee9"
PAPER_DEEP = "#e1e3dc"
INK = "#3a3d8c"          # diazo line
INK_SOFT = "#8f92c4"
INK_FAINT = "#c9cbdc"
GRAPHITE = "#24262b"
GRAPHITE_SOFT = "#5c5f68"
RED = "#c4372c"          # red pencil
OCHRE = "#a8741c"        # yellow pencil, darkened for contrast on paper
BACKGROUND = "#b3b6c0"   # the quiet majority: light graphite so the calls stay in front

# Whiteprint ramp for spectra: bare paper for quiet, violet-blue ink for loud.
DIAZO_SCALE = [
    [0.00, "#eceee9"],
    [0.18, "#d6d8e3"],
    [0.40, "#a9acd3"],
    [0.65, "#6b6fb4"],
    [0.85, "#3a3d8c"],
    [1.00, "#1b1d4f"],
]

# Cluster pencils, assigned in order of cluster id.
# Red is never a category: it only marks the selection (Detail A), revisions and outliers.
PENCILS = [GRAPHITE_SOFT, INK, "#2f7d6d", OCHRE, "#7a4f8f", "#2f6f8f", "#8a5a3a"]

FONT = "Overpass, sans-serif"
MONO = "'Overpass Mono', monospace"


def pencil(i: int) -> str:
    return PENCILS[i % len(PENCILS)]


def style_figure(fig: go.Figure, height: int, *, margin=None, show_legend=False) -> go.Figure:
    """Apply the drawing-set chart grammar to a Plotly figure."""
    axis = dict(
        showline=True, linecolor=INK, linewidth=1, mirror=True,
        ticks="outside", tickcolor=INK, ticklen=4, tickwidth=1,
        gridcolor=INK_FAINT, gridwidth=0.5, griddash="dot", zeroline=False,
        tickfont=dict(family=MONO, size=11, color=GRAPHITE_SOFT),
        title_font=dict(family=FONT, size=11, color=GRAPHITE_SOFT),
        automargin=False,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=GRAPHITE, size=12),
        margin=margin or dict(l=56, r=16, t=12, b=40),
        showlegend=show_legend,
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=PAPER, bordercolor=INK, font=dict(family=MONO, size=11, color=GRAPHITE)),
        dragmode=False,
    )
    return fig


def dimension(fig: go.Figure, y0: float, y1: float, label: str, *, color=INK, x_paper=1.012):
    """A vertical dimension line just outside the plot's right edge, spanning y0..y1, labelled to its right.

    Kept outside the data so it never hides the calls it names; labels are short to fit a narrow margin.
    """
    fig.add_shape(type="line", xref="paper", yref="y", x0=x_paper, x1=x_paper, y0=y0, y1=y1,
                  line=dict(color=color, width=1))
    for y in (y0, y1):
        fig.add_shape(type="line", xref="paper", yref="y", x0=x_paper - 0.006, x1=x_paper + 0.006,
                      y0=y, y1=y, line=dict(color=color, width=1))
    # annotations on a log axis take log10 positions; shapes take data values
    fig.add_annotation(xref="paper", yref="y", x=x_paper, xshift=6, y=math.log10((y0 * y1) ** 0.5),
                       yanchor="middle", xanchor="left", showarrow=False, text=label, align="left",
                       font=dict(family=FONT, size=10.5, color=color))


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Overpass:wght@300..900&family=Overpass+Mono:wght@300..700&display=swap');

:root {{
  --paper: {PAPER}; --paper-deep: {PAPER_DEEP}; --ink: {INK}; --ink-soft: {INK_SOFT};
  --ink-faint: {INK_FAINT}; --graphite: {GRAPHITE}; --graphite-soft: {GRAPHITE_SOFT}; --red: {RED};
  --ochre: {OCHRE};
}}

/* the sheet: a ruled double border with zone ticks, like a drawing frame */
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] {{ background: var(--paper); }}
[data-testid="stMainBlockContainer"] {{
  max-width: 1320px; padding: 2.4rem 2.6rem 3rem; margin-top: 1.2rem; margin-bottom: 1.6rem;
  border: 1.5px solid var(--ink); outline: 1px solid var(--ink-faint); outline-offset: 6px;
  background:
    repeating-linear-gradient(90deg, var(--ink) 0 1px, transparent 1px 12.5%) top / 100% 7px no-repeat,
    repeating-linear-gradient(90deg, var(--ink) 0 1px, transparent 1px 12.5%) bottom / 100% 7px no-repeat,
    var(--paper);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"] {{ display: none; }}

h1, h2, h3, h4 {{ font-family: Overpass, sans-serif !important; color: var(--graphite); letter-spacing: -0.01em; }}
p, li {{ max-width: 72ch; }}

/* sidebar = drawing index + title block */
[data-testid="stSidebar"] {{ border-right: 1.5px solid var(--ink); }}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ padding-top: 0.6rem; }}

.ds-block {{ border: 1px solid var(--ink); background: var(--paper); margin: 0 0 0.9rem; }}
.ds-block .row {{ display: grid; grid-template-columns: 5.4rem 1fr; border-top: 1px solid var(--ink-faint); }}
.ds-block .row:first-child {{ border-top: 0; }}
.ds-block .k {{ font: 600 0.62rem/1.2 Overpass, sans-serif; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink); padding: 0.45rem 0.5rem; border-right: 1px solid var(--ink-faint); }}
.ds-block .v {{ font: 400 0.82rem/1.3 Overpass, sans-serif; color: var(--graphite); padding: 0.4rem 0.55rem; }}
.ds-block .v b {{ font-weight: 800; }}
.ds-block .v a {{ color: var(--ink); text-decoration-thickness: 1px; text-underline-offset: 2px; }}
.ds-block .mono {{ font-family: 'Overpass Mono', monospace; font-size: 0.78rem; }}
.ds-project {{ font: 900 1.3rem/1.02 Overpass, sans-serif; letter-spacing: -0.015em; color: var(--graphite);
  padding: 0.7rem 0.6rem 0.6rem; text-transform: uppercase; }}
.ds-project span {{ display: block; font: 500 0.72rem/1.35 Overpass, sans-serif; letter-spacing: 0.02em;
  text-transform: none; color: var(--graphite-soft); margin-top: 0.35rem; }}

/* sheet heading inside the frame */
.ds-sheet {{ display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 1rem;
  border-bottom: 1.5px solid var(--ink); padding-bottom: 0.7rem; margin-bottom: 1.1rem; }}
.ds-sheet h1 {{ font: 900 clamp(1.7rem, 3.4vw, 2.9rem)/0.98 Overpass, sans-serif !important;
  text-transform: uppercase; letter-spacing: -0.02em; margin: 0; padding: 0; color: var(--graphite); }}
.ds-sheet p {{ margin: 0.55rem 0 0; color: var(--graphite-soft); font-size: 0.98rem; }}
.ds-sheetno {{ border: 1.5px solid var(--ink); text-align: center; min-width: 5.6rem; }}
.ds-sheetno .k {{ font: 600 0.6rem/1 Overpass, sans-serif; letter-spacing: 0.1em; color: var(--ink);
  padding: 0.35rem 0.4rem; border-bottom: 1px solid var(--ink); }}
.ds-sheetno .n {{ font: 800 1.5rem/1 'Overpass Mono', monospace; color: var(--graphite); padding: 0.35rem 0.4rem 0.3rem; }}

.ds-label {{ font: 700 0.7rem/1.2 Overpass, sans-serif; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--ink); margin: 0.2rem 0 0.25rem; display: flex; align-items: center; gap: 0.5rem; }}
.ds-label::after {{ content: ""; flex: 1; border-top: 1px solid var(--ink-faint); }}
.ds-caption {{ font-size: 0.82rem; color: var(--graphite-soft); margin: -0.2rem 0 0.6rem; }}

/* general notes, numbered as on a drawing */
.ds-notes {{ counter-reset: note; margin: 0.2rem 0 0; padding: 0; list-style: none; }}
.ds-notes li {{ counter-increment: note; display: grid; grid-template-columns: 2rem 1fr; gap: 0.2rem;
  padding: 0.5rem 0; border-top: 1px solid var(--ink-faint); font-size: 0.92rem; line-height: 1.45; }}
.ds-notes li::before {{ content: counter(note) "."; font: 700 0.85rem/1.5 'Overpass Mono', monospace; color: var(--ink); }}
.ds-notes b {{ color: var(--graphite); }}
.ds-val {{ font-family: 'Overpass Mono', monospace; color: var(--ink); font-weight: 600;
  background: none; padding: 0; white-space: nowrap; }}

/* detail callout: the selected minute */
.ds-detail-tag {{ display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; }}
.ds-detail-tag svg {{ flex: none; }}
.ds-detail-tag .t {{ font: 800 0.95rem/1.15 Overpass, sans-serif; color: var(--graphite); }}
.ds-detail-tag .s {{ font: 400 0.78rem/1.3 'Overpass Mono', monospace; color: var(--graphite-soft); }}
.ds-chip {{ display: inline-block; font: 700 0.68rem/1 Overpass, sans-serif; letter-spacing: 0.06em;
  text-transform: uppercase; padding: 0.28rem 0.45rem 0.22rem; border: 1px solid currentColor; }}

/* schedules (tables) */
.ds-schedule {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
.ds-schedule th {{ font: 700 0.64rem/1.2 Overpass, sans-serif; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink); text-align: left; border-bottom: 1.5px solid var(--ink); padding: 0.45rem 0.5rem; }}
.ds-schedule td {{ border-bottom: 1px solid var(--ink-faint); padding: 0.45rem 0.5rem; vertical-align: top; }}
.ds-schedule td.n {{ font-family: 'Overpass Mono', monospace; text-align: right; white-space: nowrap; }}
.ds-schedule tr.hi td {{ background: rgba(196,55,44,0.06); }}
.ds-swatch {{ display: inline-block; width: 0.7rem; height: 0.7rem; margin-right: 0.45rem; vertical-align: -1px; }}

/* revision table with red-pencil cloud marks */
.ds-rev td.rev {{ font: 800 0.95rem/1 'Overpass Mono', monospace; color: var(--red); text-align: center; }}


/* zone numbers across the top of the frame */
.ds-zones {{ display: grid; grid-template-columns: repeat(8, 1fr); margin: -2.4rem -2.6rem 1.9rem; height: 7px; }}
.ds-zones span {{ font: 500 0.58rem/7px 'Overpass Mono', monospace; color: var(--ink); text-align: center; }}

/* drawing index: a ruled list, not radio dots (Streamlit: radiogroup > div > label[stRadioOption]) */
[data-testid="stSidebar"] [data-testid="stRadioGroup"] {{ gap: 0; border: 1px solid var(--ink); background: var(--paper); }}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{ border-top: 1px solid var(--ink-faint); margin: 0; }}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:first-child {{ border-top: 0; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] {{ width: 100%; padding: 0.42rem 0.6rem; margin: 0; cursor: pointer; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: 0.84rem; white-space: pre; margin: 0; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: var(--paper-deep); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: var(--ink); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: var(--paper); font-weight: 700; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:focus-visible) {{ outline: 2px solid var(--red); outline-offset: -2px; }}

[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {{
  background: var(--paper); border: 1px solid var(--ink); border-radius: 0; }}

/* phone-only pieces */
.st-key-mobile_nav, .st-key-mobile_foot {{ display: none; }}

/* widgets in the drawing grammar */
[data-testid="stSegmentedControl"] button, [data-testid="stBaseButton-segmented_control"],
[data-testid="stBaseButton-segmented_controlActive"] {{ font-family: Overpass, sans-serif; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.72rem; border-radius: 0 !important; }}
[data-testid="stBaseButton-segmented_controlActive"] {{ background: var(--ink) !important; color: var(--paper) !important;
  border-color: var(--ink) !important; }}
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {{
  border-radius: 0; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.74rem; }}
[data-testid="stBaseButton-primary"] {{ background: var(--ink); border-color: var(--ink); }}
[data-testid="stBaseButton-primary"]:hover {{ background: #2c2f73; border-color: #2c2f73; }}
.stButton > button:focus-visible, [data-baseweb="select"]:focus-within {{ outline: 2px solid var(--red); outline-offset: 2px; }}
[data-testid="stAudio"] audio {{ width: 100%; filter: grayscale(1) contrast(1.05); }}
[data-testid="stImage"] img {{ border: 0; }}

/* a slow line-draw on first paint: the sheet's one authored motion */
/* only the rule under the sheet title is drawn in; text is never hidden, even if animation stalls */
.ds-sheet {{ position: relative; border-bottom-color: transparent; }}
.ds-sheet::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: -1.5px; height: 1.5px;
  background: var(--ink); transform-origin: left; animation: ds-draw 1.1s cubic-bezier(0.16, 1, 0.3, 1) both; }}
@keyframes ds-draw {{ from {{ transform: scaleX(0.04); }} to {{ transform: scaleX(1); }} }}
@media (prefers-reduced-motion: reduce) {{ .ds-sheet::after {{ animation: none; }} }}

@media (max-width: 760px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 1.4rem 0.9rem 2rem; margin: 0.4rem 0.2rem; outline: 0; }}
  .ds-zones {{ display: none; }}
  [data-testid="stMainBlockContainer"] {{ padding-top: 2.6rem; }}
  .ds-sheet {{ grid-template-columns: 1fr auto; align-items: start; gap: 0.6rem; margin-bottom: 0.8rem; }}
  .ds-sheet h1 {{ font-size: 1.55rem !important; }}
  .ds-sheet p {{ font-size: 0.88rem; }}
  .ds-sheetno {{ min-width: 3.6rem; }}
  .ds-sheetno .k {{ font-size: 0.6rem; padding: 0.25rem; }}
  .ds-sheetno .n {{ font-size: 1rem; padding: 0.25rem; }}
  .st-key-mobile_nav, .st-key-mobile_foot {{ display: block; }}
  /* on S-1 the hearing comes first: Detail A above the elevation */
  .st-key-s1_main > div > [data-testid="stHorizontalBlock"] {{ flex-direction: column-reverse; }}
}}
</style>
"""

def revision_cloud(text: str, color: str = RED) -> str:
    """A scalloped revision cloud drawn around a short label, as on a revised drawing."""
    w, h, r = max(64, int(6.4 * len(text)) + 26), 26, 4
    nx, ny = max(2, round((w - 2 * r) / (2 * r))), 2
    sx, sy = (w - 2 * r) / nx, (h - 2 * r) / ny
    d = f"M{r},{r} "
    d += "".join(f"a{sx / 2:.2f},{r} 0 0,1 {sx:.2f},0 " for _ in range(nx))
    d += "".join(f"a{r},{sy / 2:.2f} 0 0,1 0,{sy:.2f} " for _ in range(ny))
    d += "".join(f"a{sx / 2:.2f},{r} 0 0,1 {-sx:.2f},0 " for _ in range(nx))
    d += "".join(f"a{r},{sy / 2:.2f} 0 0,1 0,{-sy:.2f} " for _ in range(ny))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{text}" '
            f'style="vertical-align:middle"><path d="{d}Z" fill="none" stroke="{color}" stroke-width="1.2"/>'
            f'<text x="{w / 2}" y="{h / 2 + 4}" text-anchor="middle" font-family="Overpass" font-weight="700" '
            f'font-size="10.5" fill="{color}">{text}</text></svg>')


def delta(letter: str, color: str = RED) -> str:
    """Revision delta: the revision letter in a triangle, placed beside what it changed."""
    return (f'<svg width="20" height="18" viewBox="0 0 20 18" role="img" aria-label="revision {letter}" '
            f'style="vertical-align:-3px;margin-left:.3rem"><path d="M10 1 L19 17 L1 17 Z" fill="none" '
            f'stroke="{color}" stroke-width="1.2"/><text x="10" y="15" text-anchor="middle" font-family="Overpass" '
            f'font-weight="800" font-size="9" fill="{color}">{letter}</text></svg>')


# Detail callout bubble: a circle with letter over sheet number, as drawings reference details.
def detail_bubble(letter: str, sheet: str, color: str = RED) -> str:
    return (f'<svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true">'
            f'<circle cx="22" cy="22" r="20" fill="none" stroke="{color}" stroke-width="1.5"/>'
            f'<line x1="4" y1="22" x2="40" y2="22" stroke="{color}" stroke-width="1"/>'
            f'<text x="22" y="18" text-anchor="middle" font-family="Overpass" font-weight="800" '
            f'font-size="12" fill="{color}">{letter}</text>'
            f'<text x="22" y="34" text-anchor="middle" font-family="Overpass Mono" font-size="9" '
            f'fill="{color}">{sheet}</text></svg>')
