"""Tests run fully offline on synthetic data. Run: pytest -q"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from oceansound import pipeline
from oceansound.data import stream_local_wavs
from oceansound.features import band_level_db, extract
from oceansound.synthetic import make_segment

FS = 16_000


def test_band_level_of_pure_tone():
    """A 0.5-amplitude tone has power 0.125 -> -9.03 dB, all inside its band."""
    t = np.arange(FS * 10) / FS
    x = 0.5 * np.sin(2 * np.pi * 100 * t)
    from scipy import signal

    f, p = signal.welch(x, fs=FS, nperseg=FS)
    assert abs(band_level_db(f, p, 90, 110) - 10 * np.log10(0.125)) < 0.2
    assert band_level_db(f, p, 1000, 2000) < -60


def test_calibration_offset_shifts_levels():
    x = make_segment("ambient", FS, 10, np.random.default_rng(0))
    a = extract(x, FS).features
    b = extract(x, FS, calibration_db=100).features
    assert abs((b["b_10_100"] - a["b_10_100"]) - 100) < 1e-6
    assert abs(b["fin_excess"] - a["fin_excess"]) < 1e-6   # relative features unchanged


def test_features_respond_to_sources():
    rng = np.random.default_rng(1)
    amb = extract(make_segment("ambient", FS, 60, rng), FS).features
    fin = extract(make_segment("fin_whale", FS, 60, rng), FS).features
    blue = extract(make_segment("blue_whale", FS, 60, rng), FS).features
    ship = extract(make_segment("ship", FS, 60, rng), FS).features
    assert fin["fin_excess"] > amb["fin_excess"] + 6
    assert blue["blue_excess"] > amb["blue_excess"] + 6
    assert ship["low_minus_high_db"] > amb["low_minus_high_db"] + 10


def test_synthetic_end_to_end_recovers_classes():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        pipeline.main(["synthetic", "--n-per-class", "8", "--seg-seconds", "30", "--out", str(out)])
        s = json.loads((out / "summary.json").read_text())
        assert s["k"] == 5
        assert s["adjusted_rand_index"] > 0.9
        for name in ["ltsa", "band_levels", "pca_clusters", "silhouette", "cluster_examples"]:
            assert (out / "figures" / f"{name}.png").exists()


def test_local_wav_reader_windows():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "TEST-20200101T000000Z.wav"
        sf.write(p, make_segment("ambient", FS, 20), FS, subtype="PCM_24")
        segs = list(stream_local_wavs([p], n_segments=4, seg_seconds=5))
        assert len(segs) == 4
        assert all(s.samples.size == 5 * FS for s in segs)
        assert segs[0].start_time.year == 2020
        assert segs[-1].meta["offset_s"] == 15.0


def test_zero_filled_gap_is_rejected_and_skipped():
    """MBARI fills recording gaps with zeros; those windows must be skipped, not produce inf/NaN."""
    import pytest

    from oceansound.data import Segment
    from oceansound.features import RecordingGap

    with pytest.raises(RecordingGap):
        extract(np.zeros(FS * 10, dtype=np.float32), FS)

    good = make_segment("ambient", FS, 10, np.random.default_rng(0))
    segs = [Segment(f"s{i}", "t", None, FS, x) for i, x in
            enumerate([good, np.zeros_like(good), good])]
    df, ltsa, thumbs = pipeline.extract_all(segs)
    assert list(df["segment_id"]) == ["s0", "s2"]
    assert np.isfinite(df[pipeline.feature_columns(df)].to_numpy(dtype=float)).all()
