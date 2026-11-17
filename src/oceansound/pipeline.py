"""Command-line pipeline: segments -> features -> clusters -> outputs.

Examples
--------
# Offline demo with labelled synthetic data (no internet needed)
python -m oceansound.pipeline synthetic --n-per-class 20 --out outputs/synthetic

# See which days exist in MBARI's archive for a month
python -m oceansound.pipeline list --year 2018 --month 11

# Real data: 48 one-minute windows per day from three days
python -m oceansound.pipeline mbari --dates 2018-11-01 2018-11-02 2018-11-03 --out outputs/mbari_nov2018

# Local WAV files
python -m oceansound.pipeline local --wavs data/*.wav --out outputs/local

# Re-cluster an existing run with a fixed k
python -m oceansound.pipeline recluster --out outputs/mbari_nov2018 --k 5
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import plots
from .clustering import cluster_features, describe_clusters, evaluate_against_labels
from .config import (DEFAULT_N_SEGMENTS, DEFAULT_SEG_SECONDS, FEATURE_SETS, FEATURE_VERSION,
                     MARS_DEP02_CALIBRATION_DB, MARS_DEP02_END, MARS_DEP02_START)
from .data import Segment
from .features import RecordingGap, extract, log_freq_grid

log = logging.getLogger("oceansound")

META_COLS = ["segment_id", "source", "start_time", "label"]
# Reference metrics kept alongside the features but never used for clustering
REFERENCE_COLS = ["mbari_fin_ci", "mbari_blue_ci"]


# ----------------------------------------------------------------------------
# Core steps
# ----------------------------------------------------------------------------

class WindowErrors(RuntimeError):
    """Some windows failed for reasons other than a recording gap (possibly transient)."""


def extract_all(segments: Iterable[Segment], calibration_db: float = 0.0, strict: bool = False):
    """Run feature extraction over segments.

    Recording gaps are always skipped. Other per-window failures are skipped with a
    warning, unless ``strict``: then they raise ``WindowErrors`` so the caller can retry
    (a network hiccup must not silently remove windows from a day).
    """
    rows, ltsa, thumbs, errors = [], [], [], []
    t_start = time.time()
    for i, seg in enumerate(segments, 1):
        try:
            sf = extract(seg.samples, seg.fs, calibration_db)
        except RecordingGap as e:
            log.warning("skipping %s: %s", seg.segment_id, e)
            continue
        except Exception as e:  # noqa: BLE001 - keep going on one bad window
            log.warning("skipping %s: %s: %s", seg.segment_id, type(e).__name__, e)
            errors.append(seg.segment_id)
            continue
        rows.append({"segment_id": seg.segment_id, "source": seg.source,
                     "start_time": seg.start_time, "label": seg.label, **sf.features})
        ltsa.append(sf.ltsa_row)
        thumbs.append(sf.thumbnail)
        if i % 10 == 0:
            log.info("  %d windows processed (%.1f s)", i, time.time() - t_start)
    if errors and strict:
        raise WindowErrors(f"{len(errors)} window(s) failed: {', '.join(errors[:5])}")
    if not rows:
        raise RuntimeError("no windows were processed")
    return pd.DataFrame(rows), np.stack(ltsa), np.stack(thumbs)


def save_features(out: Path, df: pd.DataFrame, ltsa: np.ndarray, thumbs: np.ndarray, tag: str = ""):
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"features{tag}.csv", index=False)
    np.savez_compressed(out / f"spectra{tag}.npz", freqs=log_freq_grid(), ltsa=ltsa, thumbs=thumbs,
                        segment_ids=df["segment_id"].to_numpy())


def load_features(out: Path, tag: str = ""):
    df = pd.read_csv(out / f"features{tag}.csv", parse_dates=["start_time"])
    z = np.load(out / f"spectra{tag}.npz", allow_pickle=True)
    return df, z["ltsa"], z["thumbs"], z["freqs"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = (set(META_COLS) | set(REFERENCE_COLS) | {"cluster"}
            | {c for c in df.columns if c.startswith("PC")})
    return [c for c in df.columns if c not in drop]


def units_for(calibration_db: float) -> str:
    return "dB re 1 µPa" if calibration_db else "dB re FS"


def analyse(out: Path, seg_seconds: float, k: int | None = None, seed: int = 0,
            feature_set: str = "all", clip: float | None = None, calibration_db: float = 0.0) -> dict:
    """Cluster saved features and write tables, figures and a summary JSON."""
    df, ltsa, thumbs, freqs = load_features(out)
    order = np.argsort(df["start_time"].to_numpy(), kind="stable")
    df, ltsa, thumbs = df.iloc[order].reset_index(drop=True), ltsa[order], thumbs[order]

    all_cols = feature_columns(df)
    cols = FEATURE_SETS[feature_set] or all_cols
    # Safety net: drop windows with non-finite features (e.g. features cached
    # before gap detection existed) instead of crashing in StandardScaler.
    finite = np.isfinite(df[all_cols].to_numpy(dtype=float)).all(axis=1)
    n_dropped = int((~finite).sum())
    if n_dropped:
        log.warning("dropping %d windows with non-finite features: %s", n_dropped,
                    ", ".join(df.loc[~finite, "segment_id"].astype(str)))
        df, ltsa, thumbs = df[finite].reset_index(drop=True), ltsa[finite], thumbs[finite]
    res = cluster_features(df, cols, k=k, seed=seed, clip_quantile=clip)
    df["cluster"] = res.labels
    for i in range(min(3, res.pca_coords.shape[1])):
        df[f"PC{i + 1}"] = res.pca_coords[:, i]

    df.to_csv(out / "features_clustered.csv", index=False)
    refs = [c for c in REFERENCE_COLS if c in df.columns and df[c].notna().any()]
    describe_clusters(df, all_cols + refs).round(3).to_csv(out / "cluster_summary.csv")
    res.loadings.round(3).to_csv(out / "pca_loadings.csv")

    summary = {
        "n_windows": int(len(df)),
        "n_dropped_nonfinite": n_dropped,
        "seg_seconds": seg_seconds,
        "k": int(res.k),
        "silhouette_by_k": {str(kk): round(v, 4) for kk, v in res.silhouette_by_k.items()},
        "pca_explained_variance": [round(float(v), 4) for v in res.explained_variance],
        "feature_version": FEATURE_VERSION,
        "calibration_db": calibration_db,
        "units": units_for(calibration_db),
        "feature_set": feature_set,
        "clip_quantile": clip,
        "features": cols,
        "time_range": [str(df["start_time"].min()), str(df["start_time"].max())],
    }

    has_labels = df["label"].notna().all()
    if has_labels:
        ari, table = evaluate_against_labels(df["label"], df["cluster"])
        table.to_csv(out / "confusion_true_vs_cluster.csv")
        summary["adjusted_rand_index"] = round(ari, 4)
        log.info("ARI vs ground truth: %.3f\n%s", ari, table)

    fig = out / "figures"
    units = units_for(calibration_db)
    plots.plot_ltsa(ltsa, freqs, df["start_time"], fig / "ltsa.png", units=units)
    plots.plot_band_levels(df, fig / "band_levels.png", units=units)
    plots.plot_pca(res.pca_coords, res.labels, fig / "pca_clusters.png",
                   df["label"].to_numpy() if has_labels else None)
    plots.plot_silhouette(res.silhouette_by_k, res.k, fig / "silhouette.png")
    plots.plot_cluster_examples(thumbs, freqs, res.labels, seg_seconds, fig / "cluster_examples.png",
                                seed=seed)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("k=%d, silhouette=%.3f -> %s", res.k, res.silhouette_by_k.get(res.k, float("nan")), out)
    return summary


# ----------------------------------------------------------------------------
# Sub-commands
# ----------------------------------------------------------------------------

def cmd_synthetic(a):
    from .synthetic import synthetic_segments

    segs = synthetic_segments(a.n_per_class, seconds=a.seg_seconds, seed=a.seed)
    save_features(a.out, *extract_all(segs))
    analyse(a.out, a.seg_seconds, a.k, a.seed, a.features, a.clip)


def _cache_is_current(path: Path) -> bool:
    """A cached day is reused only if it has every column the current code writes.

    Features added later (e.g. the MBARI call indices) would otherwise be silently
    missing for cached days and NaN after concatenation.
    """
    if not path.exists():
        return False
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(REFERENCE_COLS) - header
    if missing:
        log.info("%s: cache predates %s, recomputing", path.name, ", ".join(sorted(missing)))
    return not missing


def _stream_day_with_retries(stream_day, date, a, attempts: int = 3, wait_s: float = 10.0):
    """Stream and extract one day; retry transient network errors, skip missing/empty days.

    Returns (features, ltsa, thumbs), or None if the day could not be processed.
    """
    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        try:
            # strict until the last attempt: a failed window (not a gap) triggers a retry of the day
            return extract_all(stream_day(date, a.n_segments, a.seg_seconds), a.calibration_db,
                               strict=not last)
        except WindowErrors as e:
            log.warning("%s: attempt %d: %s, retrying in %.0f s", date, attempt, e, wait_s * attempt)
            time.sleep(wait_s * attempt)
            continue
        except (FileNotFoundError, RuntimeError) as e:   # day missing, or every window was a gap
            log.warning("%s: skipped (%s)", date, e)
            return None
        except Exception as e:  # noqa: BLE001 - network errors come from s3fs/botocore/aiohttp
            if attempt == attempts:
                log.warning("%s: failed after %d attempts (%s: %s)", date, attempts, type(e).__name__, e)
                return None
            log.warning("%s: attempt %d failed (%s), retrying in %.0f s", date, attempt,
                        type(e).__name__, wait_s * attempt)
            time.sleep(wait_s * attempt)


def cmd_mbari(a):
    from .data import stream_mbari_day

    if a.calibrated:
        bad = [d for d in a.dates if not (MARS_DEP02_START <= d <= MARS_DEP02_END)]
        if bad:
            raise SystemExit(f"--calibrated uses the 2017-06-13..2024-08-07 hydrophone calibration; "
                             f"outside that period: {bad}")
        a.calibration_db = MARS_DEP02_CALIBRATION_DB
    cache = a.out / "cache"
    parts, failed = [], []
    for d in a.dates:
        date = dt.date.fromisoformat(d)
        tag = (f"_{date:%Y%m%d}" + (f"_cal{a.calibration_db:g}" if a.calibration_db else "")
               + f"_v{FEATURE_VERSION}")
        if _cache_is_current(cache / f"features{tag}.csv") and not a.force:
            log.info("%s: using cache", date)
        else:
            log.info("%s: streaming %d x %gs windows from S3", date, a.n_segments, a.seg_seconds)
            result = _stream_day_with_retries(stream_mbari_day, date, a)
            if result is None:
                failed.append(str(date))
                continue
            save_features(cache, *result, tag)
        parts.append(load_features(cache, tag))
    if failed:
        log.warning("days not processed: %s (re-run the same command to retry them)", ", ".join(failed))

    if not parts:
        raise SystemExit("no usable days")
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    save_features(a.out, df, np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))
    analyse(a.out, a.seg_seconds, a.k, a.seed, a.features, a.clip, a.calibration_db)


def cmd_local(a):
    from .data import stream_local_wavs

    segs = stream_local_wavs([Path(p) for p in a.wavs], a.n_segments, a.seg_seconds)
    save_features(a.out, *extract_all(segs, a.calibration_db))
    analyse(a.out, a.seg_seconds, a.k, a.seed, a.features, a.clip, a.calibration_db)


def cmd_list(a):
    from .data import list_mbari_files

    for key in list_mbari_files(a.year, a.month):
        print(key)


def cmd_wind(a):
    """Join buoy wind to an existing run and test the sea-state interpretation."""
    from . import environment
    from .data import fetch_ndbc_stdmet

    df = pd.read_csv(a.out / "features_clustered.csv", parse_dates=["start_time"])
    t = pd.to_datetime(df["start_time"], utc=True)
    years = range(t.min().year, t.max().year + 1)
    env = pd.concat([environment.read_ndbc_stdmet(fetch_ndbc_stdmet(a.station, y)) for y in years])
    j = environment.join_nearest(df, env)
    j.to_csv(a.out / "features_wind.csv", index=False)
    cols = [c for c in feature_columns(df)]
    corr = environment.wind_correlations(j, cols)
    corr.round(4).to_csv(a.out / "wind_correlations.csv")
    r = float(corr.loc["b_1k_8k", "spearman_r"])
    summary = json.loads((a.out / "summary.json").read_text())
    plots.plot_wind_vs_noise(j, a.out / "figures" / "wind_vs_noise.png", station=a.station, r=r,
                             units=summary.get("units", "dB re FS"))

    summary["wind"] = {
        "station": a.station,
        "n_matched": int(j["wind_ms"].notna().sum()),
        "spearman_b_1k_8k": round(r, 3),
        "wind_ms_median_by_cluster": {str(k): round(float(v), 2) for k, v in
                                      j.groupby("cluster")["wind_ms"].median().items()},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("wind vs b_1k_8k: Spearman r=%.3f (n=%d)\n%s", r, summary["wind"]["n_matched"],
             corr.head(8).round(3))


def cmd_patterns(a):
    """Monthly and month x hour summaries of selected features."""
    from . import temporal

    df = pd.read_csv(a.out / "features_clustered.csv")
    summary = json.loads((a.out / "summary.json").read_text())
    feats = temporal.PATTERN_FEATURES
    monthly = temporal.monthly_summary(df, list(feats))
    monthly.round(2).to_csv(a.out / "monthly_summary.csv")
    temporal.diel_contrast(df, list(feats)).to_csv(a.out / "diel_night_minus_day.csv")
    fig = a.out / "figures"
    plots.plot_monthly(monthly, feats, fig / "monthly.png", units=summary.get("units", "dB re FS"))
    grids = {f: temporal.month_hour_grid(df, f) for f in ["fin_excess", "blue_excess", "var_200_2k", "b_1k_8k"]}
    plots.plot_month_hour(grids, fig / "month_hour.png")
    shares = pd.crosstab(temporal.add_time_columns(df)["year_month"], df["cluster"], normalize="index")
    shares.round(3).to_csv(a.out / "cluster_share_by_month.csv")
    log.info("monthly medians:\n%s", monthly[[f"{f}_median" for f in feats] + ["n_windows"]].round(1))


def cmd_compare(a):
    """Compare fin_excess / blue_excess with MBARI's call indices (calibrated runs only)."""
    from . import validation

    df = pd.read_csv(a.out / "features_clustered.csv")
    if df[list(validation.PAIRS.values())].isna().all().all():
        raise SystemExit("this run has no MBARI call indices; re-run 'mbari' with --calibrated")
    minute = validation.minute_agreement(df)
    monthly, month_r = validation.monthly_agreement(df)
    by_cluster = validation.index_by_cluster(df)
    minute.round(4).to_csv(a.out / "mbari_ci_minute_agreement.csv", index=False)
    monthly.round(4).to_csv(a.out / "mbari_ci_monthly.csv")
    month_r.round(4).to_csv(a.out / "mbari_ci_monthly_agreement.csv", index=False)
    by_cluster.round(4).to_csv(a.out / "mbari_ci_by_cluster.csv")
    plots.plot_index_comparison(monthly, df, a.out / "figures" / "mbari_ci_comparison.png",
                                dict(zip(minute["ours"], minute["spearman_r"])),
                                dict(zip(month_r["ours"], month_r["spearman_r_months"])))
    summary = json.loads((a.out / "summary.json").read_text())
    summary["mbari_ci"] = {
        "minute_spearman": {r.ours: round(r.spearman_r, 3) for r in minute.itertuples()},
        "monthly_spearman": {r.ours: round(r.spearman_r_months, 3) for r in month_r.itertuples()},
        "median_ci_by_cluster": {str(k): {c: round(float(v), 4) for c, v in row.items() if c != "n_windows"}
                                 for k, row in by_cluster.iterrows()},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("per minute:\n%s\nper month:\n%s\nby cluster:\n%s", minute.round(3), month_r.round(3),
             by_cluster.round(3))


def cmd_recluster(a):
    if a.src:  # start a new run directory from an existing run's features
        import shutil

        a.out.mkdir(parents=True, exist_ok=True)
        for name in ["features.csv", "spectra.npz", "summary.json"]:
            shutil.copy(a.src / name, a.out / name)
    summary = json.loads((a.out / "summary.json").read_text())
    analyse(a.out, summary["seg_seconds"], a.k, a.seed, a.features, a.clip,
            summary.get("calibration_db", 0.0))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="oceansound", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, out_required=True):
        sp.add_argument("--out", type=Path, required=out_required, help="output directory")
        sp.add_argument("--k", type=int, default=None, help="fix number of clusters (default: pick by silhouette)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--features", choices=sorted(FEATURE_SETS), default="all",
                        help="feature set for clustering ('relative' is less sensitive to wind)")
        sp.add_argument("--clip", type=float, default=None,
                        help="winsorize each feature to [q, 1-q] quantiles before clustering, e.g. 0.01")

    def windows(sp):
        sp.add_argument("--n-segments", type=int, default=DEFAULT_N_SEGMENTS, help="windows per file/day")
        sp.add_argument("--seg-seconds", type=float, default=DEFAULT_SEG_SECONDS)
        sp.add_argument("--calibration-db", type=float, default=0.0,
                        help="dB offset added to all levels (0 = dB re full scale)")

    s = sub.add_parser("synthetic", help="labelled synthetic demo (offline)")
    common(s)
    s.add_argument("--n-per-class", type=int, default=20)
    s.add_argument("--seg-seconds", type=float, default=DEFAULT_SEG_SECONDS)
    s.set_defaults(func=cmd_synthetic)

    s = sub.add_parser("mbari", help="stream days from MBARI Pacific Sound (S3)")
    common(s)
    windows(s)
    s.add_argument("--dates", nargs="+", required=True, help="UTC dates, YYYY-MM-DD")
    s.add_argument("--calibrated", action="store_true",
                   help=f"report dB re 1 µPa using MBARI's deployment-02 hydrophone calibration "
                        f"(+{MARS_DEP02_CALIBRATION_DB} dB; valid {MARS_DEP02_START}..{MARS_DEP02_END})")
    s.add_argument("--force", action="store_true", help="ignore cached days")
    s.set_defaults(func=cmd_mbari)

    s = sub.add_parser("local", help="local WAV/FLAC files")
    common(s)
    windows(s)
    s.add_argument("--wavs", nargs="+", required=True)
    s.set_defaults(func=cmd_local)

    s = sub.add_parser("list", help="list MBARI daily files for a month")
    s.add_argument("--year", type=int, required=True)
    s.add_argument("--month", type=int, required=True)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("wind", help="compare a run with NOAA buoy wind (needs internet once)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--station", default="46042", help="NDBC station id (46042 = Monterey Bay)")
    s.set_defaults(func=cmd_wind)

    s = sub.add_parser("patterns", help="seasonal (monthly) and diel (hour-of-day) summaries of a run")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_patterns)

    s = sub.add_parser("compare", help="compare fin/blue features with MBARI's call indices")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("recluster", help="re-run clustering on an existing output dir")
    common(s)
    s.add_argument("--src", type=Path, default=None,
                   help="copy features from this run into --out first (keeps the source run intact)")
    s.set_defaults(func=cmd_recluster)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
