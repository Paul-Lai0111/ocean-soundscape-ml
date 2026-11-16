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


NDBC_SAMPLE = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC   mi    ft
2018 11 01 00 50 175  2.5  3.1  1.15 10.81  5.90 283 1018.8  13.3  14.0 999.0 99.0 99.00
2018 11 01 01 50 175 99.0 99.0 99.00 10.81  5.90 283 1018.8  13.3  14.0 999.0 99.0 99.00
2018 11 01 02 50 163  8.4  9.6  0.98 10.00  5.62 289 1019.0  13.4  14.1 999.0 99.0 99.00
"""


def test_ndbc_parse_and_nearest_join():
    import gzip

    import pandas as pd

    from oceansound import environment

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "46042h2018.txt.gz"
        p.write_bytes(gzip.compress(NDBC_SAMPLE.encode()))
        env = environment.read_ndbc_stdmet(p)
    assert len(env) == 2                      # 99.0 wind = missing, dropped
    assert env["wind_ms"].tolist() == [2.5, 8.4]
    windows = pd.DataFrame({"start_time": ["2018-11-01 00:40:00+00:00",   # nearest 00:50
                                           "2018-11-01 02:30:00+00:00",   # nearest 02:50
                                           "2018-11-05 00:00:00+00:00"]}) # nothing within 1 h
    j = environment.join_nearest(windows, env)
    assert j["wind_ms"].iloc[0] == 2.5 and j["wind_ms"].iloc[1] == 8.4
    assert np.isnan(j["wind_ms"].iloc[2])


def test_clipping_stops_a_single_outlier_owning_a_cluster():
    import pandas as pd

    from oceansound.clustering import cluster_features

    rng = np.random.default_rng(0)
    blobs = [rng.normal(m, 1, (50, 3)) for m in ([0, 0, 0], [6, 6, 6], [0, 6, 0])]
    X = np.vstack(blobs + [[[0, 0, 500]]])                # three groups + one extreme window
    df = pd.DataFrame(X, columns=["x", "y", "z"])
    raw = cluster_features(df, ["x", "y", "z"], k=3)
    clipped = cluster_features(df, ["x", "y", "z"], k=3, clip_quantile=0.01)
    assert min(np.bincount(raw.labels)) == 1              # outlier takes a cluster, two groups merge
    assert sorted(np.bincount(clipped.labels)) == [50, 50, 51]


def test_relative_feature_set_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        pipeline.main(["synthetic", "--n-per-class", "6", "--seg-seconds", "30", "--out", str(out),
                       "--features", "relative", "--clip", "0.01"])
        s = json.loads((out / "summary.json").read_text())
        assert s["feature_set"] == "relative" and s["features"] == [
            "fin_excess", "blue_excess", "var_10_100", "var_200_2k", "p90_minus_med_200_2k", "kurtosis"]
        # Without absolute levels, ambient and ship look alike (both lack calls and pulses),
        # so the clean synthetic classes are recovered less well than with "all" features.
        assert s["adjusted_rand_index"] > 0.5


def test_read_window_and_listening_prep_local():
    import datetime as dt

    from oceansound.data import file_offset_seconds, for_listening, read_window

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "TEST-20200101T000000Z.wav"
        x = make_segment("ambient", FS, 20, np.random.default_rng(3))
        sf.write(p, x, FS, subtype="PCM_24")
        start = dt.datetime(2020, 1, 1, 0, 0, 5, tzinfo=dt.timezone.utc)
        off = file_offset_seconds(str(p), start)
        assert off == 5.0
        y, fs = read_window(str(p), off, 3)
        assert fs == FS and y.size == 3 * FS
        assert np.allclose(y, x[5 * FS:8 * FS], atol=1e-6)
        z, play_fs = for_listening(y, fs, speedup=10)
        assert play_fs == 10 * FS and abs(np.max(np.abs(z)) - 0.9) < 1e-6


def test_mars_calibration_constant_matches_mbari_recipe():
    """MBARI: volts = x * 3, PSD_dB re uPa = 10log10(PSD_volts) - sensitivity."""
    from oceansound.config import MARS_DEP02_CALIBRATION_DB, MARS_DEP02_SENSITIVITY_DB

    x = make_segment("ambient", FS, 10, np.random.default_rng(0))
    ours = extract(x, FS, calibration_db=MARS_DEP02_CALIBRATION_DB).features["b_10_100"]
    from scipy import signal

    f, p = signal.welch((x - x.mean()).astype(np.float64) * 3, fs=FS, nperseg=FS, noverlap=FS // 2)
    ref = band_level_db(f, p, 10, 100) - MARS_DEP02_SENSITIVITY_DB
    assert abs(ours - ref) < 0.01


def test_calibrated_flag_rejects_dates_outside_deployment():
    import pytest

    with pytest.raises(SystemExit):
        pipeline.main(["mbari", "--dates", "2016-01-01", "--calibrated", "--out", "unused"])
