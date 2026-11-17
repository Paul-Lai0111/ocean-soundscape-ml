"""Compare the pipeline's call features with MBARI's published call indices.

MBARI's tutorials define a fin whale and a blue whale B-call index (CI) as a
ratio of calibrated spectrum levels at peak vs background frequencies (see
config.MBARI_CALL_INDICES). Our features ``fin_excess`` / ``blue_excess`` are
designed for the same calls but use wider bands and a dB difference. If both
track the same biology they should rank minutes, months and clusters alike.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from .temporal import add_time_columns

PAIRS = {  # our feature -> MBARI index
    "fin_excess": "mbari_fin_ci",
    "blue_excess": "mbari_blue_ci",
}


def minute_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between each of our features and its MBARI index, per minute."""
    rows = []
    for ours, ref in PAIRS.items():
        d = df[[ours, ref]].dropna()
        r, p = stats.spearmanr(d[ours], d[ref])
        rows.append({"ours": ours, "mbari": ref, "spearman_r": r, "p_value": p, "n": len(d)})
    return pd.DataFrame(rows)


def monthly_agreement(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly mean of each MBARI index (as in MBARI's tutorials) and median of our feature.

    Returns (monthly table, Spearman r across months for each pair).
    """
    d = add_time_columns(df)
    g = d.groupby("year_month")
    table = pd.DataFrame({**{f"{o}_median": g[o].median() for o in PAIRS},
                          **{f"{r}_mean": g[r].mean() for r in PAIRS.values()}})
    rows = []
    for ours, ref in PAIRS.items():
        r, p = stats.spearmanr(table[f"{ours}_median"], table[f"{ref}_mean"])
        rows.append({"ours": ours, "mbari": ref, "spearman_r_months": r, "p_value": p,
                     "n_months": len(table)})
    return table, pd.DataFrame(rows)


def index_by_cluster(df: pd.DataFrame) -> pd.DataFrame:
    """Median MBARI index per cluster: do the 'whale' clusters carry the whale signal?"""
    cols = list(PAIRS.values())
    out = df.groupby("cluster")[cols].median()
    out.insert(0, "n_windows", df.groupby("cluster").size())
    return out
