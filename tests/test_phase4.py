"""Tests for account features, detector counts, ranking helpers and precision@k."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml.anomaly import mean_rank
from aml.config import Config, PathsConfig
from aml.evaluate import precision_at_k
from aml.features import add_detector_counts, build_account_features, transform_features


def make_cfg(tmp_path: Path) -> Config:
    return Config(paths=PathsConfig(processed_dir=tmp_path, tables_dir=tmp_path, figures_dir=tmp_path))


@pytest.fixture()
def toy(tmp_path: Path) -> Config:
    rows = [
        # A -> P (1000 USD, 10:00), P -> B (990, 11:00): P is a conduit
        ("A", "P", "2022-09-01 10:00", 1000.0, "ACH", "US Dollar", "US Dollar"),
        ("P", "B", "2022-09-01 11:00", 990.0, "ACH", "US Dollar", "US Dollar"),
        ("A", "C", "2022-09-02 10:00", 200.0, "Cheque", "US Dollar", "Euro"),
        ("A", "A", "2022-09-02 12:00", 50.0, "Reinvestment", "US Dollar", "US Dollar"),
    ]
    df = pd.DataFrame(rows, columns=["src", "dst", "timestamp", "amt_usd", "payment_format",
                                     "pay_ccy", "recv_ccy"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["txn_id"] = np.arange(len(df), dtype="int64")
    df["amt_paid"] = df["amt_usd"]
    df.to_parquet(tmp_path / "transactions.parquet", index=False)
    return make_cfg(tmp_path)


def test_account_features_values(toy: Config) -> None:
    f = build_account_features(toy)
    assert set(f.index) == {"A", "P", "B", "C"}  # self-loop-only accounts excluded
    a, p = f.loc["A"], f.loc["P"]
    assert a["deg_out"] == 2 and a["n_out"] == 2 and a["n_self_loops"] == 1
    assert a["usd_out_sum"] == pytest.approx(1200.0) and a["usd_out_max"] == pytest.approx(1000.0)
    assert p["ratio_out_in"] == pytest.approx(0.99)
    assert p["median_dwell_hours"] == pytest.approx(1.0)
    assert p["has_both_directions"] == 1 and f.loc["B", "has_both_directions"] == 0
    assert f.loc["C", "cross_ccy_share"] == pytest.approx(1.0)
    assert a["share_fmt_ach"] == pytest.approx(0.5)
    assert f.notna().all().all()


def test_detector_counts_and_transform(toy: Config) -> None:
    f = build_account_features(toy)
    findings = pd.DataFrame({"detector": ["fan_out", "temporal_cycle", "fan_out"],
                             "account_ids": [["A", "P"], ["A", "B"], ["P"]]})
    out = add_detector_counts(f, findings)
    assert out.loc["A", "hits_fan_out"] == 1 and out.loc["A", "hits_temporal_cycle"] == 1
    assert out.loc["P", "hits_fan_out"] == 2 and out.loc["C", "hits_pass_through"] == 0
    # idempotent
    assert add_detector_counts(out, findings).shape == out.shape
    x, cols = transform_features(out)
    assert x.shape == (4, len(cols)) and np.isfinite(x).all()
    assert np.allclose(x.mean(axis=0), 0, atol=1e-9)


def test_mean_rank_orders_consistently() -> None:
    s = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    r = mean_rank(s)
    assert list(r) == pytest.approx([1 / 3, 2 / 3, 1.0])
    tied = mean_rank(pd.DataFrame({"a": [1.0, 1.0, 5.0]}))
    assert tied.iloc[0] == tied.iloc[1] < tied.iloc[2]


def test_precision_at_k_with_and_without_ties() -> None:
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert precision_at_k(y, s, 1) == 1.0
    assert precision_at_k(y, s, 3) == pytest.approx(2 / 3)
    # everything tied: expected precision equals the base rate, whatever k
    assert precision_at_k(y, np.zeros(5), 2) == pytest.approx(0.4)
    # k larger than n is clipped
    assert precision_at_k(y, s, 50) == pytest.approx(0.4)
