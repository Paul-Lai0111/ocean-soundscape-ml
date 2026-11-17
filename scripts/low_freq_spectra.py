"""Save exact 1 Hz calibrated spectra (5-60 Hz) for every window of a run.

Used to investigate the fin whale index (docs/analysis_zh.md, section 8). The
pipeline's LTSA is interpolated onto a log grid; call-index comparisons need the
same 1 Hz Welch bins MBARI uses. Re-streams each minute listed in a run's
features.csv, so it needs internet (about 20 minutes for the year run).

Usage: python scripts/low_freq_spectra.py outputs/mbari_2018-19_hourly
Writes <run>/low_freq_spectra.npz with freqs (Hz), psd_db (n x f) and segment_ids.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from oceansound.config import MARS_DEP02_CALIBRATION_DB
from oceansound.data import file_offset_seconds

run = Path(sys.argv[1])
df = pd.read_csv(run / "features.csv", parse_dates=["start_time"])
summary_cal = MARS_DEP02_CALIBRATION_DB
out = run / "low_freq_spectra.npz"

import soundfile as sf  # noqa: E402

from oceansound.data import _s3  # noqa: E402

rows, ids, t0 = [], [], time.time()
# open each daily file once and read all of its minutes (re-opening costs ~1 s per minute)
for n_day, (source, day) in enumerate(df.groupby("source", sort=False), 1):
    for attempt in range(3):
        try:
            day_rows, day_ids = [], []
            with _s3().open(source, "rb", block_size=2**22) as fh, sf.SoundFile(fh) as snd:
                fs = snd.samplerate
                for r in day.itertuples():
                    snd.seek(int(round(file_offset_seconds(source, r.start_time.to_pydatetime()) * fs)))
                    x = snd.read(60 * fs, dtype="float32")
                    if x.size < 60 * fs:
                        raise OSError(f"short read for {r.segment_id}")
                    x = x.astype(np.float64) - x.mean()
                    # identical to features.extract: Welch, 1 s Hann, 50% overlap, 1 Hz bins
                    f, psd = signal.welch(x, fs=fs, nperseg=fs, noverlap=fs // 2, window="hann")
                    keep = (f >= 5) & (f <= 60)
                    day_rows.append((10 * np.log10(np.maximum(psd[keep], 1e-20)) + summary_cal).astype(np.float32))
                    day_ids.append(r.segment_id)
            break
        except Exception as e:  # noqa: BLE001 - transient network errors
            print(f"retry {source}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        raise SystemExit(f"could not read {source}")
    rows += day_rows
    ids += day_ids
    print(f"day {n_day}: {len(ids)}/{len(df)} windows ({time.time() - t0:.0f} s)", flush=True)

np.savez_compressed(out, freqs=f[keep], psd_db=np.stack(rows), segment_ids=np.array(ids))
print("wrote", out, flush=True)
