"""Seasonal and diel (time-of-day) patterns of soundscape features.

Long sampled runs (e.g. one minute per hour for a year) are summarised as
monthly distributions and as a month x local-hour grid. Medians are used
throughout because single loud events (ships, earthquakes) skew means.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "America/Los_Angeles"   # Monterey Bay; handles PST/PDT

# Features summarised by default, with a short reading for plots and reports
PATTERN_FEATURES: dict[str, str] = {
    "fin_excess": "Fin whale 20 Hz band above neighbours (dB)",
    "blue_excess": "Blue whale 40-50 Hz band above neighbours (dB)",
    "var_200_2k": "200-2000 Hz variability (humpback song proxy, dB)",
    "b_1k_8k": "1-8 kHz level (wind / sea state)",
    "b_10_100": "10-100 Hz level (shipping, large whales)",
}


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add month (1-12), year-month label and local hour of day."""
    d = df.copy()
    t = pd.to_datetime(d["start_time"], utc=True)
    local = t.dt.tz_convert(LOCAL_TZ)
    d["month"] = t.dt.month
    d["year_month"] = t.dt.strftime("%Y-%m")
    d["local_hour"] = local.dt.hour
    return d


def monthly_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Median and interquartile range per year-month, plus window counts."""
    d = add_time_columns(df)
    g = d.groupby("year_month")
    parts = {"n_windows": g.size()}
    for f in features:
        parts[f"{f}_median"] = g[f].median()
        parts[f"{f}_q25"] = g[f].quantile(0.25)
        parts[f"{f}_q75"] = g[f].quantile(0.75)
    return pd.DataFrame(parts)


def month_hour_grid(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Median of ``feature`` for each (year-month, local hour) cell."""
    d = add_time_columns(df)
    return d.pivot_table(index="year_month", columns="local_hour", values=feature, aggfunc="median")


def diel_contrast(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Night (20-04 local) minus day (08-16 local) median, per month and feature (dB)."""
    d = add_time_columns(df)
    night = d[(d["local_hour"] >= 20) | (d["local_hour"] < 4)]
    day = d[(d["local_hour"] >= 8) & (d["local_hour"] < 16)]
    return (night.groupby("year_month")[features].median()
            - day.groupby("year_month")[features].median()).round(2)
