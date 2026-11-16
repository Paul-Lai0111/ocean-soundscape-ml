"""Synthetic soundscape generator with known labels.

Used to validate the pipeline end to end without internet access and to
check that the features separate source types the way we expect. The
signals are simplified caricatures, not physically accurate models:

- ambient   : pink-ish background noise (wind / distant sources)
- ship      : strong low-frequency broadband noise + propeller tonals,
              amplitude-modulated at the blade rate
- fin_whale : ~1 s downswept pulses (~23 -> 17 Hz) every ~12 s
- blue_whale: long (~10 s) tonal near 45 Hz with a slight downsweep
- humpback  : frequency-modulated sweeps between ~200 and 1500 Hz
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from scipy import signal

from .data import Segment

CLASSES = ("ambient", "ship", "fin_whale", "blue_whale", "humpback")


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise via spectral shaping of white noise."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=float)
    f[0] = 1.0
    spec /= np.sqrt(f)
    x = np.fft.irfft(spec, n)
    return x / np.std(x)


def _lowpass_noise(n: int, fs: int, fc: float, rng: np.random.Generator) -> np.ndarray:
    sos = signal.butter(4, fc, btype="low", fs=fs, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n))
    return x / np.std(x)


def _chirp_pulse(fs: int, dur: float, f0: float, f1: float) -> np.ndarray:
    t = np.arange(int(dur * fs)) / fs
    return signal.chirp(t, f0=f0, f1=f1, t1=dur, method="linear") * signal.windows.tukey(t.size, 0.3)


def _place(x: np.ndarray, pulse: np.ndarray, start: int, amp: float) -> None:
    end = min(x.size, start + pulse.size)
    if start < x.size:
        x[start:end] += amp * pulse[: end - start]


def make_segment(kind: str, fs: int = 16_000, seconds: float = 60.0,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate one window of synthetic audio of the given kind."""
    rng = rng or np.random.default_rng()
    n = int(fs * seconds)
    t = np.arange(n) / fs
    x = _pink_noise(n, rng) * 0.05 * 10 ** (rng.uniform(-3, 3) / 20)  # ambient, +-3 dB

    if kind == "ambient":
        pass

    elif kind == "ship":
        amp = 0.6 * 10 ** (rng.uniform(-6, 6) / 20)
        blade_rate = rng.uniform(6, 12)                       # Hz
        broadband = _lowpass_noise(n, fs, rng.uniform(300, 600), rng)
        modulation = 1 + 0.5 * np.sin(2 * np.pi * blade_rate * t)
        tonals = sum(np.sin(2 * np.pi * k * blade_rate * t + rng.uniform(0, 2 * np.pi)) / k
                     for k in range(1, 8))
        x += amp * (broadband * modulation + 0.3 * tonals)

    elif kind == "fin_whale":
        amp = 0.6 * 10 ** (rng.uniform(-4, 4) / 20)
        pulse = _chirp_pulse(fs, 1.0, 23, 17)
        ipi = rng.uniform(10, 14)                             # inter-pulse interval, s
        for s in np.arange(rng.uniform(0, ipi), seconds - 1, ipi):
            _place(x, pulse, int(s * fs), amp)

    elif kind == "blue_whale":
        amp = 0.4 * 10 ** (rng.uniform(-4, 4) / 20)
        f0 = rng.uniform(44, 47)
        call = _chirp_pulse(fs, 10.0, f0, f0 - 1.5)
        for s in np.arange(rng.uniform(0, 20), seconds - 10, rng.uniform(20, 30)):
            _place(x, call, int(s * fs), amp)

    elif kind == "humpback":
        amp = 0.15 * 10 ** (rng.uniform(-4, 4) / 20)
        s = rng.uniform(0, 2)
        while s < seconds - 2:
            dur = rng.uniform(0.4, 2.0)
            f0, f1 = rng.uniform(200, 1500, size=2)
            _place(x, _chirp_pulse(fs, dur, f0, f1), int(s * fs), amp)
            s += dur + rng.uniform(0.5, 3.0)

    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {CLASSES}")

    return x.astype(np.float32)


def synthetic_segments(n_per_class: int = 20, fs: int = 16_000, seconds: float = 60.0,
                       seed: int = 0, classes: tuple[str, ...] = CLASSES):
    """Yield a shuffled set of labelled synthetic segments."""
    rng = np.random.default_rng(seed)
    kinds = [k for k in classes for _ in range(n_per_class)]
    rng.shuffle(kinds)
    t0 = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    for i, kind in enumerate(kinds):
        yield Segment(
            segment_id=f"synth_{i:04d}",
            source="synthetic",
            start_time=t0 + dt.timedelta(minutes=30 * i),
            fs=fs,
            samples=make_segment(kind, fs, seconds, rng),
            label=kind,
        )
