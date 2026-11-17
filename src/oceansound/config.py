"""Shared constants: frequency bands, defaults, and dataset locations.

Frequency bands are chosen around well-known sound sources in the
NE Pacific (Monterey Bay). They are deliberately broad and overlapping;
PCA downstream takes care of the redundancy.
"""

from __future__ import annotations

import math

# --- MBARI Pacific Sound (MARS cabled observatory, Monterey Bay) -------------
# Public AWS Open Data buckets, no credentials needed (anonymous access).
# 16 kHz is the decimated product: one WAV file per UTC day.
MBARI_BUCKET_16K = "pacific-sound-16khz"
MBARI_BUCKET_2K = "pacific-sound-2khz"

# --- Analysis defaults --------------------------------------------------------
DEFAULT_FS = 16_000          # Hz
DEFAULT_SEG_SECONDS = 60     # length of each analysed window
DEFAULT_N_SEGMENTS = 48      # windows per day (48 -> one per 30 min)

FMIN = 10.0                  # Hz, lowest frequency analysed
FMAX = 7_900.0               # Hz, just under Nyquist for 16 kHz

# Long-term spectral average (LTSA) and thumbnail spectrogram resolution
N_LOG_FREQ_BINS = 128
THUMB_TIME_FRAMES = 120

# --- Frequency bands (Hz) -----------------------------------------------------
# name: (f_low, f_high, short description)
BANDS: dict[str, tuple[float, float, str]] = {
    "b_10_100":     (10, 100,    "Low-frequency band: distant shipping, large whales"),
    "b_100_1k":     (100, 1000,  "Mid band: nearby ships, humpback song (lower part)"),
    "b_1k_8k":      (1000, 7900, "High band: wind/surface noise, humpback, dolphins"),
    "fin_17_25":    (17, 25,     "Fin whale 20 Hz pulse band"),
    "blue_40_50":   (40, 50,     "NE Pacific blue whale B-call fundamental (~43-47 Hz)"),
    "hump_200_2k":  (200, 2000,  "Core humpback song band"),
}

# Flanking bands used to measure *excess* energy in a narrow band
# (band level minus mean of its neighbours). Tonal/biological calls stand out,
# broadband noise (ships, wind) does not.
EXCESS_BANDS: dict[str, tuple[tuple[float, float], list[tuple[float, float]]]] = {
    "fin_excess":  ((17, 25), [(11, 15), (28, 34)]),
    "blue_excess": ((40, 50), [(32, 37), (55, 62)]),
}

# --- Feature sets for clustering ---------------------------------------------
# "all": every scalar feature. On real data its main axis is wind/sea state,
#        because absolute levels and spectral shape all rise with wind
#        (Spearman |r| 0.66-0.84 vs NOAA buoy wind, Nov 2018).
# "relative": features that compare a band with its own neighbours or with
#        itself over time, so a uniform rise of the noise floor cancels out.
#        Wind dependence of the clusters drops from eta^2 0.38 to 0.06.
FEATURE_SETS: dict[str, list[str] | None] = {
    "all": None,
    "relative": ["fin_excess", "blue_excess", "var_10_100", "var_200_2k",
                 "p90_minus_med_200_2k", "kurtosis"],
}

# --- Calibration (MBARI MARS hydrophone, deployment 02) ---------------------
# Source: MBARI_MARS_Hydrophone_Deployment02.json (bitbucket.org/mbari/pacific-sound),
# referenced in every 16 kHz WAV header from 2017-06 to 2024-08.
# Instrument: Ocean Sonics icListen HF SN 1689, Reson TC4059-1 element, 890 m depth.
# Lab calibration 2017-02-02: -177.9 dB re V/uPa at 250 Hz, -176.8 at 10 kHz.
# MBARI's tutorial converts float samples to volts with x * 3, then removes the
# sensitivity. In dB that is +20*log10(3) - sensitivity. We use the 250 Hz value,
# which MBARI also applies below 250 Hz (where the whale bands are); across 10 Hz-8 kHz
# the sensitivity varies by at most 1.1 dB, so the constant is within ~0.55 dB.
MARS_DEP02_START = "2017-06-13"
MARS_DEP02_END = "2024-08-07"
MARS_DEP02_SENSITIVITY_DB = -177.9          # dB re 1 V/uPa at 250 Hz
MARS_VOLTS_PER_UNIT = 3.0                   # float sample -> volts (MBARI tutorial)
MARS_DEP02_CALIBRATION_DB = round(20 * math.log10(MARS_VOLTS_PER_UNIT)
                                  - MARS_DEP02_SENSITIVITY_DB, 2)   # = 187.44

# --- MBARI call indices (reference metrics, not used for clustering) --------
# From MBARI's Pacific Sound tutorials (J. Ryan): CI = mean spectrum level (dB re
# 1 uPa^2/Hz) at the peak frequencies divided by the mean at the background
# frequencies, from 60 s Welch spectra with a 1 s Hann window (1 Hz bins).
# CI = 1 means no signal. Needs calibrated levels, so it is NaN for uncalibrated runs.
#   name: ((peak Hz, ...), (background Hz, ...))
MBARI_CALL_INDICES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "mbari_fin_ci": ((20, 21), (12, 34)),    # fin whale 20 Hz pulse
    "mbari_blue_ci": ((43, 44), (37, 50)),   # blue whale B call, 3rd harmonic
}
