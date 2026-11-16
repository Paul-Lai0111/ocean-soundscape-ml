"""Environmental context for soundscape windows: wind and waves from NOAA buoys.

Wind-driven breaking waves are the dominant source of ocean noise above ~1 kHz
when no ships or animals are close (the "Wenz curves"). If the pipeline's
high-frequency features track buoy wind speed, the "sea-state" interpretation
of the clusters is supported by independent data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# NDBC uses 99 / 999 / 9999 as missing-value codes
_MISSING = {"WSPD": 99.0, "GST": 99.0, "WVHT": 99.0, "DPD": 99.0, "WDIR": 999.0}


def read_ndbc_stdmet(path: Path) -> pd.DataFrame:
    """Parse an NDBC historical stdmet file into a UTC-indexed frame (wind_ms, gust_ms, wave_m)."""
    df = pd.read_csv(path, sep=r"\s+", skiprows=[1], na_values=None)
    df = df.rename(columns={"#YY": "YY"})
    t = pd.to_datetime(dict(year=df["YY"], month=df["MM"], day=df["DD"], hour=df["hh"], minute=df["mm"]),
                       utc=True)
    out = pd.DataFrame({"time": t})
    for src, dst in [("WSPD", "wind_ms"), ("GST", "gust_ms"), ("WVHT", "wave_m")]:
        v = df[src].astype(float)
        out[dst] = v.where(v < _MISSING[src])
    return out.dropna(subset=["wind_ms"]).sort_values("time").reset_index(drop=True)


def join_nearest(df: pd.DataFrame, env: pd.DataFrame, tolerance: str = "60min") -> pd.DataFrame:
    """Attach the nearest buoy observation (within ``tolerance``) to each window."""
    left = df.copy()
    left["start_time"] = pd.to_datetime(left["start_time"], utc=True)
    left["_t"] = left["start_time"] + pd.Timedelta(seconds=30)    # window centre
    left = left.sort_values("_t")
    merged = pd.merge_asof(left, env, left_on="_t", right_on="time",
                           direction="nearest", tolerance=pd.Timedelta(tolerance))
    return merged.drop(columns=["_t", "time"]).sort_index()


def wind_correlations(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Spearman correlation of each feature with wind speed (rank-based: robust to outliers)."""
    d = df.dropna(subset=["wind_ms"])
    rows = []
    for f in features:
        r, p = stats.spearmanr(d["wind_ms"], d[f])
        rows.append({"feature": f, "spearman_r": r, "p_value": p, "n": len(d)})
    return pd.DataFrame(rows).set_index("feature").sort_values("spearman_r", key=np.abs, ascending=False)
