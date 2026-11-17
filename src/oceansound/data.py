"""Data access: stream windows from MBARI's public S3 archive or local WAVs.

The MBARI 16 kHz product is one WAV file per UTC day (well over 1 GB), so we
never download a whole file. ``soundfile`` can read from any seekable file
object, and ``s3fs`` turns S3 byte-range requests into a seekable file, so
reading a 60 s window only transfers that window (~3 MB).
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from .config import DEFAULT_N_SEGMENTS, DEFAULT_SEG_SECONDS, MBARI_BUCKET_16K


@dataclass
class Segment:
    """One analysis window of single-channel audio."""

    segment_id: str
    source: str                 # file name / S3 key / "synthetic"
    start_time: dt.datetime     # UTC start of the window
    fs: int                     # sample rate (Hz)
    samples: np.ndarray         # float32, shape (n,)
    label: str | None = None    # ground truth (synthetic data only)
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# MBARI Pacific Sound on S3
# ----------------------------------------------------------------------------

def _s3():
    import s3fs  # imported lazily so the synthetic demo works without it

    return s3fs.S3FileSystem(anon=True)


@functools.lru_cache(maxsize=64)
def _list_month(year: int, month: int, bucket: str) -> tuple[str, ...]:
    return tuple(sorted(_s3().glob(f"{bucket}/{year:04d}/{month:02d}/*.wav")))


def list_mbari_files(year: int, month: int, bucket: str = MBARI_BUCKET_16K) -> list[str]:
    """List the daily WAV files available for a given month.

    Returns S3 keys like ``pacific-sound-16khz/2018/11/MARS-20181101T000000Z-16kHz.wav``.
    We glob instead of hard-coding the name so small naming changes don't break us.
    """
    return list(_list_month(year, month, bucket))   # cached: one listing per month per process


def find_mbari_file(date: dt.date, bucket: str = MBARI_BUCKET_16K) -> str:
    """Find the S3 key of the daily file for ``date``."""
    stamp = date.strftime("%Y%m%d")
    for key in list_mbari_files(date.year, date.month, bucket):
        if stamp in Path(key).name:
            return key
    raise FileNotFoundError(f"No file for {date} in s3://{bucket}/{date:%Y/%m}/")


def _file_start_time(name: str) -> dt.datetime | None:
    """Parse a start time like ``20181101T000000Z`` out of a file name."""
    m = re.search(r"(\d{8}T\d{6})Z?", name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)


def _evenly_spaced_offsets(n_frames: int, fs: int, seg_seconds: float, n_segments: int) -> list[int]:
    seg_len = int(seg_seconds * fs)
    if n_frames <= seg_len:
        return [0]
    n_segments = max(1, n_segments)
    step = (n_frames - seg_len) / max(1, n_segments - 1) if n_segments > 1 else 0
    return [int(round(i * step)) for i in range(n_segments)]


def _read_windows(fileobj, source: str, seg_seconds: float, n_segments: int,
                  t0: dt.datetime | None) -> Iterator[Segment]:
    with sf.SoundFile(fileobj) as snd:
        fs = snd.samplerate
        seg_len = int(seg_seconds * fs)
        for i, off in enumerate(_evenly_spaced_offsets(snd.frames, fs, seg_seconds, n_segments)):
            snd.seek(off)
            x = snd.read(seg_len, dtype="float32", always_2d=True)[:, 0]  # first channel
            if x.size < min(seg_len, snd.frames - off):
                # A dropped connection can surface as a short read rather than an exception.
                raise OSError(f"short read in {source} at {off / fs:.0f} s: {x.size} of {seg_len} samples")
            start = (t0 or dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)) + dt.timedelta(seconds=off / fs)
            yield Segment(
                segment_id=f"{Path(source).stem}_{i:03d}",
                source=source,
                start_time=start,
                fs=fs,
                samples=x,
                meta={"offset_s": off / fs},
            )


def stream_mbari_day(date: dt.date, n_segments: int = DEFAULT_N_SEGMENTS,
                     seg_seconds: float = DEFAULT_SEG_SECONDS,
                     bucket: str = MBARI_BUCKET_16K) -> Iterator[Segment]:
    """Yield ``n_segments`` evenly spaced windows from one day of MARS audio."""
    key = find_mbari_file(date, bucket)
    t0 = _file_start_time(Path(key).name) or dt.datetime.combine(date, dt.time(), dt.timezone.utc)
    with _s3().open(key, "rb", block_size=2**22) as f:
        yield from _read_windows(f, key, seg_seconds, n_segments, t0)


# ----------------------------------------------------------------------------
# Local files
# ----------------------------------------------------------------------------

def stream_local_wavs(paths: list[Path], n_segments: int = DEFAULT_N_SEGMENTS,
                      seg_seconds: float = DEFAULT_SEG_SECONDS) -> Iterator[Segment]:
    """Yield evenly spaced windows from each local audio file."""
    for p in paths:
        p = Path(p)
        yield from _read_windows(str(p), p.name, seg_seconds, n_segments, _file_start_time(p.name))


# ----------------------------------------------------------------------------
# NOAA NDBC buoy data (wind, waves) for environmental context
# ----------------------------------------------------------------------------

NDBC_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station}h{year}.txt.gz"


def fetch_ndbc_stdmet(station: str, year: int, cache_dir: Path = Path("data/ndbc")) -> Path:
    """Download (once) a year of NDBC standard meteorological data; return the local path.

    Station 46042 (Monterey Bay, ~27 nm WNW of Monterey) is the closest long-running
    buoy to the MARS hydrophone.
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{station}h{year}.txt.gz"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(NDBC_URL.format(station=station, year=year), tmp)
        tmp.rename(path)
    return path


def file_offset_seconds(source: str, start_time: dt.datetime) -> float:
    """Seconds from the start of ``source`` (parsed from its name) to ``start_time``."""
    t0 = _file_start_time(Path(source).name)
    if t0 is None:
        raise ValueError(f"cannot parse a start time from {source!r}")
    return (start_time - t0).total_seconds()


def read_window(source: str, offset_s: float, seconds: float) -> tuple[np.ndarray, int]:
    """Re-read one window of audio for playback.

    ``source`` is either an S3 key from the MBARI bucket (as stored in features.csv)
    or a local file path. Returns (samples as float32, sample rate).
    """
    local = Path(source)
    if local.exists():
        opener = lambda: open(local, "rb")  # noqa: E731
    elif source.startswith(("pacific-sound-", "s3://")):
        opener = lambda: _s3().open(source.removeprefix("s3://"), "rb", block_size=2**22)  # noqa: E731
    else:
        raise FileNotFoundError(f"audio for {source!r} is not available (synthetic or moved file)")
    with opener() as f, sf.SoundFile(f) as snd:
        fs = snd.samplerate
        snd.seek(max(0, int(round(offset_s * fs))))
        x = snd.read(int(seconds * fs), dtype="float32", always_2d=True)[:, 0]
    return x, fs


def for_listening(x: np.ndarray, fs: int, speedup: float = 1.0) -> tuple[np.ndarray, int]:
    """Prepare audio for human listening: remove DC, peak-normalize, optionally speed up.

    Playing ``speedup`` times faster raises every frequency by that factor, so the
    20 Hz fin whale pulses become 200 Hz at 10x - the usual way to listen to
    large-whale calls on ordinary speakers.
    """
    y = x.astype(np.float64) - float(np.mean(x))
    peak = float(np.max(np.abs(y))) or 1.0
    return (0.9 * y / peak).astype(np.float32), int(round(fs * speedup))
