"""Tests for leakage-free transaction features and the supervised helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml.config import Config, PathsConfig
from aml.features import TXN_FEATURES_FILE, build_transaction_features
from aml.supervised import best_f1_threshold, feature_columns, metrics_at, temporal_split


def day(d: float) -> pd.Timestamp:
    return pd.Timestamp("2022-09-01") + pd.Timedelta(days=d)


@pytest.fixture(scope="module")
def feats(tmp_path_factory) -> pd.DataFrame:
    tmp = tmp_path_factory.mktemp("p5")
    rows = [  # src, dst, time, usd, ccy pair, label
        ("A", "B", day(0.0), 100.0, "Euro", "Euro", 0),     # 0: first ever
        ("A", "B", day(0.0), 50.0, "Euro", "Euro", 0),      # 1: same minute as txn 0
        ("A", "C", day(2.0), 200.0, "Euro", "US Dollar", 0),  # 2: A has 2 prior out (0,1)
        ("X", "A", day(2.5), 70.0, "Euro", "Euro", 0),      # 3: A has prior out 3 total
        ("A", "B", day(10.0), 10.0, "Euro", "Euro", 1),     # 4: 1d window empty; 7d empty
        ("A", "A", day(10.0), 5.0, "Euro", "Euro", 0),      # 5: self-loop
    ]
    df = pd.DataFrame(rows, columns=["src", "dst", "timestamp", "amt_usd", "pay_ccy", "recv_ccy",
                                     "is_laundering"])
    df["txn_id"] = np.arange(len(df), dtype="int64")
    df["amt_paid"] = df["amt_usd"]
    df["payment_format"] = "ACH"
    df.to_parquet(tmp / "transactions.parquet", index=False)
    cfg = Config(paths=PathsConfig(processed_dir=tmp, tables_dir=tmp, figures_dir=tmp))
    build_transaction_features(cfg)
    return pd.read_parquet(tmp / TXN_FEATURES_FILE).set_index("txn_id").sort_index()


def test_strictly_prior_history(feats: pd.DataFrame) -> None:
    # txns 0 and 1 share a minute: neither sees the other, nor itself
    assert feats.loc[0, "snd_out_cnt_30d"] == 0 and feats.loc[1, "snd_out_cnt_30d"] == 0
    assert feats.loc[0, "pair_seen_before"] == 0 and feats.loc[1, "pair_seen_before"] == 0
    # txn 2 (day 2): both earlier A->B payments are inside 7d and 30d windows, not the 1d window
    assert feats.loc[2, "snd_out_cnt_1d"] == 0
    assert feats.loc[2, "snd_out_cnt_7d"] == 2 and feats.loc[2, "snd_out_cnt_30d"] == 2
    assert feats.loc[2, "snd_out_usd_7d"] == pytest.approx(150.0)
    assert feats.loc[2, "snd_out_dcp_7d"] == 1       # both went to B
    assert feats.loc[2, "pair_seen_before"] == 0     # A->C is new
    # txn 3 (X->A, day 2.5): receiver A has prior *outgoing* history and a prior-out count of 3
    assert feats.loc[3, "rcv_out_cnt_7d"] == 3
    assert feats.loc[3, "rcv_in_cnt_30d"] == 0
    assert feats.loc[3, "snd_in_cnt_30d"] == 0       # X was never paid


def test_window_expiry_and_pair_history(feats: pd.DataFrame) -> None:
    # txn 4 (day 10): the day-0/2/2.5 payments are older than 7 days but inside 30 days
    assert feats.loc[4, "snd_out_cnt_7d"] == 0
    assert feats.loc[4, "snd_out_cnt_30d"] == 3
    assert feats.loc[4, "pair_seen_before"] == 1
    assert feats.loc[4, "snd_out_xc_share_30d"] == pytest.approx(1 / 3)  # one cross-ccy of three
    assert feats.loc[4, "rcv_in_cnt_30d"] == 2       # B received txns 0 and 1


def test_own_features_and_label_carried(feats: pd.DataFrame) -> None:
    assert feats.loc[2, "is_cross_ccy"] == 1 and feats.loc[0, "is_cross_ccy"] == 0
    assert feats.loc[5, "is_self_loop"] == 1
    assert feats.loc[0, "log_amt_usd"] == pytest.approx(np.log1p(100.0))
    assert feats["is_laundering"].tolist() == [0, 0, 0, 0, 1, 0]
    assert "is_laundering" not in feature_columns(feats.reset_index())
    assert feats.notna().all().all() if "pair_seen_before" in feats else True


def test_temporal_split_and_threshold() -> None:
    df = pd.DataFrame({"timestamp": pd.date_range("2022-09-01", periods=10, freq="D")})
    codes, b = temporal_split(df, 0.6, 0.2)
    assert codes.tolist() == [0] * 6 + [1] * 2 + [2] * 2
    assert b["val_start"] == df["timestamp"].iloc[6] and b["test_start"] == df["timestamp"].iloc[8]
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    thr, f1 = best_f1_threshold(y, p)
    assert f1 == pytest.approx(1.0) and 0.3 < thr <= 0.7
    m = metrics_at(y, p, 0.5, (2,))
    assert m["tp"] == 3 and m["fp"] == 0 and m["precision@2"] == 1.0
