"""Build the compact demo runs in demo/ that the hosted dashboard reads.

Real-data runs under outputs/ are large (the year's spectra.npz is ~47 MB) and are
not committed. This copies the tables of selected runs and stores their spectra
compactly: LTSA as float16 and the per-window spectrogram thumbnails as 8-bit
values (128 frequencies x 60 one-second frames; quantization error <= 0.13 dB).
Cluster names, and the evidence behind each, are attached for viewers.

Usage (after the runs exist): python scripts/make_demo_data.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"

RUNS = [
    {
        "src": "outputs/mbari_2018-19_relative",
        "dst": "1_year_2018-19",
        "title": "One year, Jul 2018 - Jun 2019 (2,189 minutes)",
        "description": (
            "One minute per hour, every 4th day, calibrated to dB re 1 µPa. Clustered on the six "
            "'relative' features (wind-insensitive). The song-season cluster holds about half of "
            "all minutes in October-November and 0.5-3% in March-August."
        ),
        "cluster_names": {
            "0": ["Background", "Most minutes; MBARI fin and blue indices near 1 (no call)."],
            "1": ["Whale song season", "Fin, blue and humpback indicators all elevated; highest "
                                       "MBARI fin (1.091) and blue (1.050) indices."],
            "2": ["Impulsive transients", "Very high kurtosis; short pulses, several suspected "
                                          "instrument transients."],
        },
    },
    {
        "src": "outputs/mbari_nov2018_cal_relative",
        "dst": "2_week_2018-11",
        "title": "One week, 1-7 Nov 2018 (314 minutes)",
        "description": (
            "One minute every 30 minutes, calibrated. The four clusters were named from their "
            "spectrograms; the blue whale name is backed by MBARI's blue whale call index. "
            "The loudest minute (2 Nov 16:19 UTC) is an M3.3 earthquake 78 km away."
        ),
        "cluster_names": {
            "0": ["Background", "Fin whale chorus, wind and ship noise."],
            "1": ["Humpback song", "200-2000 Hz variability 3.6 vs 1.2-1.3 elsewhere; named from "
                                   "spectrograms (no MBARI humpback index)."],
            "2": ["Blue whale B calls", "blue_excess 10.2 dB vs 3.3-6.2; highest MBARI blue "
                                        "whale index of the week (1.178)."],
            "3": ["Impulsive transients", "High kurtosis; broadband pulses."],
        },
    },
]


def compact_spectra(src: Path, dst: Path) -> dict:
    z = np.load(src / "spectra.npz", allow_pickle=True)
    thumbs = z["thumbs"].astype(np.float32)                       # (n, 128 freqs, 120 frames)
    lo, hi = (float(v) for v in np.percentile(thumbs, [0.5, 99.9]))
    n, nf, nt = thumbs.shape
    halved = thumbs.reshape(n, nf, nt // 2, 2).mean(axis=-1)      # 120 -> 60 frames (1 s each)
    q = np.clip(np.round((halved - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    np.savez_compressed(dst / "spectra.npz", freqs=z["freqs"], ltsa=z["ltsa"].astype(np.float16),
                        thumbs=q, thumb_db_range=np.array([lo, hi]), segment_ids=z["segment_ids"])
    return {"thumb_db_range": [round(lo, 2), round(hi, 2)], "max_quantization_error_db": round((hi - lo) / 510, 3)}


def main():
    DEMO.mkdir(exist_ok=True)
    for spec in RUNS:
        src, dst = ROOT / spec["src"], DEMO / spec["dst"]
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir()
        df = pd.read_csv(src / "features_clustered.csv")
        df.round(4).to_csv(dst / "features_clustered.csv", index=False)
        shutil.copy(src / "cluster_summary.csv", dst / "cluster_summary.csv")
        info = compact_spectra(src, dst)
        summary = json.loads((src / "summary.json").read_text())
        summary["demo"] = {k: spec[k] for k in ("title", "description", "cluster_names")}
        summary["demo"].update(source_run=spec["src"], **info)
        (dst / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        size = sum(p.stat().st_size for p in dst.iterdir()) / 1e6
        print(f"{dst.relative_to(ROOT)}: {len(df)} windows, {size:.1f} MB")


if __name__ == "__main__":
    main()
