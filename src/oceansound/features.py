"""Acoustic feature extraction for one window of audio.

For each window we compute:

1. A power spectral density (Welch, 1 Hz resolution) -> band levels,
   "excess" levels for narrow biological bands, spectral shape.
2. A short-time spectrogram -> how much band energy fluctuates over time
   (pulsed calls fluctuate, ship/wind noise is steady), plus a small
   log-frequency thumbnail for the dashboard.
3. Waveform statistics (kurtosis = impulsiveness).

Levels are in dB relative to digital full scale unless a calibration offset is
given. To report absolute levels (dB re 1 uPa^2/Hz) you need the hydrophone's
sensitivity and the recorder gain; see README "Calibration".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

from .config import BANDS, EXCESS_BANDS, FMAX, FMIN, MBARI_CALL_INDICES, N_LOG_FREQ_BINS, THUMB_TIME_FRAMES

EPS = 1e-20

# Recording gaps in the MBARI archive are filled with exact zeros (e.g. ~10 h on
# 2018-11-07). Real 24-bit hydrophone audio is essentially never exactly 0, so
# a window with more than this fraction of zero samples is treated as a gap.
MAX_ZERO_FRACTION = 0.01


class RecordingGap(ValueError):
    """Raised for windows that are (mostly) zero-filled recording gaps."""


@dataclass
class SegmentFeatures:
    features: dict[str, float]     # scalar features for clustering
    ltsa_row: np.ndarray           # PSD (dB) on the log-frequency grid, shape (N_LOG_FREQ_BINS,)
    thumbnail: np.ndarray          # spectrogram (dB), shape (N_LOG_FREQ_BINS, THUMB_TIME_FRAMES)


def log_freq_grid(n: int = N_LOG_FREQ_BINS, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """Log-spaced frequency centres used for the LTSA and thumbnails."""
    return np.geomspace(fmin, fmax, n)


def _db(p):
    return 10.0 * np.log10(np.maximum(p, EPS))


def _band_mask(f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (f >= lo) & (f < hi)


def band_level_db(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Total power in [lo, hi) Hz, in dB (integrated PSD)."""
    m = _band_mask(f, lo, hi)
    df = f[1] - f[0]
    return float(_db(psd[m].sum() * df))


def mean_psd_db(f: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Mean spectral density in [lo, hi) Hz, in dB/Hz."""
    return float(_db(psd[_band_mask(f, lo, hi)].mean()))


def extract(x: np.ndarray, fs: int, calibration_db: float = 0.0) -> SegmentFeatures:
    """Compute features for one window.

    Parameters
    ----------
    x : 1-D float array of samples
    fs : sample rate in Hz
    calibration_db : added to every level (0 -> dB re full scale)
    """
    x = np.asarray(x, dtype=np.float64)
    zero_frac = float(np.mean(x == 0)) if x.size else 1.0
    if zero_frac > MAX_ZERO_FRACTION:
        raise RecordingGap(f"{zero_frac:.0%} of samples are exactly zero (recording gap)")
    x = x - x.mean()
    feats: dict[str, float] = {}

    # ---- 1. Welch PSD, 1 Hz resolution -------------------------------------
    f, psd = signal.welch(x, fs=fs, nperseg=fs, noverlap=fs // 2, window="hann")
    psd = psd * 10 ** (calibration_db / 10)
    in_range = _band_mask(f, FMIN, FMAX)

    feats["broadband_db"] = band_level_db(f, psd, FMIN, FMAX)
    for name, (lo, hi, _) in BANDS.items():
        feats[name] = band_level_db(f, psd, lo, hi)

    for name, ((lo, hi), flanks) in EXCESS_BANDS.items():
        flank = np.mean([mean_psd_db(f, psd, a, b) for a, b in flanks])
        feats[name] = mean_psd_db(f, psd, lo, hi) - flank

    # Spectral shape over the analysed range
    fr, pr = f[in_range], psd[in_range]
    feats["centroid_log10_hz"] = float(np.log10(np.sum(fr * pr) / np.sum(pr)))
    feats["flatness_db"] = float(_db(np.exp(np.mean(np.log(pr + EPS))) / np.mean(pr)))
    # Low-to-high balance: how "red" the spectrum is
    feats["low_minus_high_db"] = feats["b_10_100"] - feats["b_1k_8k"]

    # ---- 2. Short-time spectrogram ------------------------------------------
    nper = 4096
    fs_s, ts, Z = signal.stft(x, fs=fs, nperseg=nper, noverlap=nper // 2, window="hann")
    S = (np.abs(Z) ** 2) * 10 ** (calibration_db / 10)        # power, (freq, time)

    def frame_levels(lo, hi):
        return _db(S[_band_mask(fs_s, lo, hi)].sum(axis=0))

    # Temporal variability (std of per-frame band level, dB). Pulsed calls -> high.
    feats["var_10_100"] = float(np.std(frame_levels(10, 100)))
    feats["var_200_2k"] = float(np.std(frame_levels(200, 2000)))
    # Peak-to-median per-frame level: transient prominence
    lv = frame_levels(200, 2000)
    feats["p90_minus_med_200_2k"] = float(np.percentile(lv, 90) - np.median(lv))

    # ---- 3. Waveform statistics ---------------------------------------------
    feats["kurtosis"] = float(stats.kurtosis(x, fisher=True))

    # ---- MBARI call indices (reference only; need calibrated dB re 1 uPa) -----
    psd_db = _db(psd)
    for name, (peak, background) in MBARI_CALL_INDICES.items():
        if calibration_db:
            pk = np.mean([psd_db[np.argmin(np.abs(f - hz))] for hz in peak])
            bg = np.mean([psd_db[np.argmin(np.abs(f - hz))] for hz in background])
            feats[name] = float(pk / bg)
        else:
            feats[name] = float("nan")

    # ---- LTSA row and thumbnail on a shared log-frequency grid -------------
    grid = log_freq_grid()
    ltsa_row = np.interp(grid, f, _db(psd)).astype(np.float32)

    S_db = _db(S)
    S_log = np.stack([np.interp(grid, fs_s, S_db[:, j]) for j in range(S_db.shape[1])], axis=1)
    # Average consecutive frames down to a fixed width
    edges = np.linspace(0, S_log.shape[1], THUMB_TIME_FRAMES + 1).astype(int)
    thumb = np.stack(
        [S_log[:, a:max(b, a + 1)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1
    ).astype(np.float16)

    return SegmentFeatures(features=feats, ltsa_row=ltsa_row, thumbnail=thumb)
