"""Shared constants: frequency bands, defaults, and dataset locations.

Frequency bands are chosen around well-known sound sources in the
NE Pacific (Monterey Bay). They are deliberately broad and overlapping;
PCA downstream takes care of the redundancy.
"""

from __future__ import annotations

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
