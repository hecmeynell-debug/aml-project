"""Detector tests on a synthetic fixture with known positives and negatives."""

import numpy as np
import pandas as pd
import pytest

from aml.config import DetectorsConfig
from aml.detectors import (
    FINDING_COLUMNS, fan_in, fan_out, gather_scatter, pass_through, scatter_gather,
    temporal_cycles,
)

T0 = pd.Timestamp("2022-09-01 00:00")


def hours(h: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=h)


# (src, dst, time, usd)
ROWS = [
    # valid temporal cycle V1->V2->V3->V1, increasing times, amounts within 20%
    ("V1", "V2", hours(1), 1000.0), ("V2", "V3", hours(2), 950.0), ("V3", "V1", hours(3), 900.0),
    # wrong order: W1->W2 at 5h, W2->W3 at 4h (earlier!), W3->W1 at 6h  -> no time-ordered loop
    ("W1", "W2", hours(5), 1000.0), ("W2", "W3", hours(4), 1000.0), ("W3", "W1", hours(6), 1000.0),
    # amounts break tolerance: X1->X2->X3->X1 but last hop is 3x the first
    ("X1", "X2", hours(7), 1000.0), ("X2", "X3", hours(8), 1000.0), ("X3", "X1", hours(9), 3000.0),
    # loop too slow: Y cycle spans 30 days (> 14 day window)
    ("Y1", "Y2", hours(10), 1000.0), ("Y2", "Y3", hours(10 + 24 * 15), 1000.0),
    ("Y3", "Y1", hours(10 + 24 * 30), 1000.0),
]
# fan-out: F sends to 8 distinct accounts within a few hours
ROWS += [("F", f"FO{i}", hours(20 + i * 0.5), 100.0 + i) for i in range(8)]
# fan-in: 8 distinct accounts pay G
ROWS += [(f"FI{i}", "G", hours(30 + i * 0.5), 100.0 + i) for i in range(8)]
# scatter-gather: S -> I1..I3 -> D with later second hops
ROWS += [("S", f"I{i}", hours(40 + i), 1000.0) for i in range(3)]
ROWS += [(f"I{i}", "D", hours(50 + i), 990.0) for i in range(3)]
# gather-scatter: 3 sources -> H -> 3 destinations
ROWS += [(f"GS{i}", "H", hours(60 + i), 500.0) for i in range(3)]
ROWS += [("H", f"GD{i}", hours(70 + i), 495.0) for i in range(3)]
# pass-through P: receives 2 x 5000, forwards ~all within 1 hour; plus a retaining account R
ROWS += [("PA", "P", hours(80), 5000.0), ("PB", "P", hours(90), 5000.0),
         ("P", "PC", hours(81), 4990.0), ("P", "PD", hours(91), 4995.0)]
ROWS += [("RA", "R", hours(80), 5000.0), ("RB", "R", hours(90), 5000.0),
         ("R", "RC", hours(200), 100.0), ("R", "RD", hours(210), 100.0)]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    out = pd.DataFrame(ROWS, columns=["src", "dst", "timestamp", "amt_usd"])
    out["txn_id"] = np.arange(len(out), dtype="int64")
    out["pay_ccy"] = out["recv_ccy"] = "US Dollar"
    return out


@pytest.fixture(scope="module")
def cfg() -> DetectorsConfig:
    return DetectorsConfig(fan_percentile=90.0, fan_min_degree=5, fan_windows_days=(1, 7))


def accounts(findings: pd.DataFrame) -> set[str]:
    return {a for ids in findings["account_ids"] for a in ids}


def test_columns(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    assert list(temporal_cycles(df, cfg).columns) == FINDING_COLUMNS
    assert list(fan_out(df, cfg).columns) == FINDING_COLUMNS


def test_valid_cycle_found_and_bad_ones_rejected(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    f = temporal_cycles(df, cfg)
    assert len(f) == 1
    row = f.iloc[0]
    assert set(row["account_ids"]) == {"V1", "V2", "V3"}
    assert row["txn_ids"] == [0, 1, 2]
    assert accounts(f).isdisjoint({"W1", "X1", "Y1"})  # wrong order / amounts / too slow
    assert 0.5 <= row["score"] <= 1.0


def test_cycle_length_bound(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    short = DetectorsConfig(**{**cfg.__dict__, "cycle_length_bound": 2})
    assert temporal_cycles(df, short).empty  # the only valid cycle has 3 hops


def test_cycle_cap_is_enforced(df: pd.DataFrame, cfg: DetectorsConfig, caplog) -> None:
    capped = DetectorsConfig(**{**cfg.__dict__, "max_cycles_per_component": 0})
    with caplog.at_level("WARNING"):
        temporal_cycles(df, capped)
    assert any("cap hit" in r.message for r in caplog.records)


def test_fan_out_and_fan_in(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    fo, fi = fan_out(df, cfg), fan_in(df, cfg)
    assert set(a[0] for a in fo["account_ids"]) == {"F"}
    assert set(a[0] for a in fi["account_ids"]) == {"G"}
    assert len(fo.iloc[0]["txn_ids"]) == 8
    assert fo.iloc[0]["score"] > 0


def test_scatter_gather(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    f = scatter_gather(df, cfg)
    assert len(f) == 1
    assert f.iloc[0]["account_ids"][0] == "S" and f.iloc[0]["account_ids"][-1] == "D"
    assert {"I0", "I1", "I2"} <= set(f.iloc[0]["account_ids"])
    assert len(f.iloc[0]["txn_ids"]) == 6


def test_gather_scatter(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    f = gather_scatter(df, cfg)
    assert [a[0] for a in f["account_ids"]] == ["H"]


def test_pass_through_flags_conduit_only(df: pd.DataFrame, cfg: DetectorsConfig) -> None:
    f = pass_through(df, cfg)
    flagged = {a[0] for a in f["account_ids"]}
    assert "P" in flagged
    assert "R" not in flagged  # retains almost everything


def test_typology_recall_and_precision_tables() -> None:
    from aml.evaluate import cycle_recall_by_hops, detector_precision, typology_recall

    findings = pd.DataFrame({
        "detector": ["temporal_cycle", "fan_out"],
        "txn_ids": [[1, 2, 3], [10]],
    })
    patterns = pd.DataFrame({
        "attempt_id": [0, 0, 0, 0, 1, 1],
        "typology": ["CYCLE"] * 4 + ["FAN-OUT"] * 2,
        "detail": ["Max 3 hops"] * 4 + ["Max 2-degree Fan-Out"] * 2,
        "txn_id": [1, 2, 3, 4, 10, 11],
    })
    rec = typology_recall(findings, patterns, threshold=0.5)
    assert rec.loc["CYCLE", "temporal_cycle"] == 1.0      # 3 of 4 txns >= 50%
    assert rec.loc["FAN-OUT", "temporal_cycle"] == 0.0
    assert rec.loc["FAN-OUT", "fan_out"] == 1.0           # 1 of 2 txns >= 50%
    assert rec.loc["ALL", "any_detector"] == 1.0
    labels = pd.Series({1: 1, 2: 1, 3: 0, 4: 1, 10: 0, 11: 1})
    prec = detector_precision(findings, labels, patterns)
    assert prec.loc["temporal_cycle", "precision"] == pytest.approx(2 / 3)
    assert prec.loc["fan_out", "n_laundering"] == 0
    assert cycle_recall_by_hops(findings, patterns).loc["3", "detected"] == 1.0
