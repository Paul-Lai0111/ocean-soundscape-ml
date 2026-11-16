"""Data access: stream windows from MBARI's public S3 archive or local WAVs.

The MBARI 16 kHz product is one WAV file per UTC day (well over 1 GB), so we
never download a whole file. ``soundfile`` can read from any seekable file
object, and ``s3fs`` turns S3 byte-range requests into a seekable file, so
reading a 60 s window only transfers that window (~3 MB).
"""

from __future__ import annotations

import datetime as dt
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


def list_mbari_files(year: int, month: int, bucket: str = MBARI_BUCKET_16K) -> list[str]:
    """List the daily WAV files available for a given month.

    Returns S3 keys like ``pacific-sound-16khz/2018/11/MARS-20181101T000000Z-16kHz.wav``.
    We glob instead of hard-coding the name so small naming changes don't break us.
    """
    fs = _s3()
    return sorted(fs.glob(f"{bucket}/{year:04d}/{month:02d}/*.wav"))


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
