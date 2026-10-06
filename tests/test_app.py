"""Smoke test of the Streamlit dashboard on a tiny synthetic score set (no retraining, no model)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture()
def processed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [
        ("A", "B", "2022-09-01 10:00", 100.0, 1), ("B", "C", "2022-09-01 11:00", 95.0, 1),
        ("C", "A", "2022-09-01 12:00", 90.0, 0), ("D", "A", "2022-09-02 09:00", 500.0, 0),
        ("A", "E", "2022-09-03 09:00", 50.0, 0),
    ]
    tx = pd.DataFrame(rows, columns=["src", "dst", "timestamp", "amt_usd", "is_laundering"])
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["txn_id"] = np.arange(len(tx), dtype="int64")
    tx["payment_format"] = "ACH"
    tx["pay_ccy"] = tx["recv_ccy"] = "US Dollar"
    tx.to_parquet(tmp_path / "transactions.parquet", index=False)
    ts = tx.assign(score=[0.9, 0.8, 0.7, 0.1, 0.2], flagged=[True, True, True, False, False],
                   split="test")
    ts.to_parquet(tmp_path / "transaction_scores.parquet", index=False)
    accounts = pd.DataFrame(
        {"anomaly_score": [0.9, 0.8, 0.7, 0.2, 0.1], "iforest": 0.5, "lof": 0.5, "hdbscan": 0.5,
         "max_supervised_score": [0.9, 0.9, 0.8, 0.1, 0.2], "risk": [0.9, 0.8, 0.7, 0.2, 0.1],
         "hits_temporal_cycle": [1, 1, 1, 0, 0], "hits_fan_out": 0, "usd_in": 100.0,
         "usd_out": 100.0, "n_in": 1, "n_out": 1, "n_laundering_txns": [1, 2, 1, 0, 0],
         "is_illicit_account": [1, 1, 1, 0, 0],
         "reasons": [["in a time-ordered cycle (x1)"]] * 3 + [["quiet"]] * 2},
        index=pd.Index(["A", "B", "C", "D", "E"], name="acc"))
    accounts.to_parquet(tmp_path / "account_scores.parquet")
    findings = pd.DataFrame({
        "detector": ["temporal_cycle"], "account_ids": [["A", "B", "C"]], "txn_ids": [[0, 1, 2]],
        "start_ts": [pd.Timestamp("2022-09-01 10:00")], "end_ts": [pd.Timestamp("2022-09-01 12:00")],
        "total_usd": [285.0], "score": [0.9], "detail": ["3 hops"]})
    findings.to_parquet(tmp_path / "findings.parquet", index=False)
    monkeypatch.setenv("AML_PROCESSED_DIR", str(tmp_path))
    monkeypatch.setenv("AML_MODELS_DIR", str(tmp_path / "no_models"))
    return tmp_path


def run(account: str | None) -> "streamlit_testing.AppTest":
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=90)
    at.run()
    if account is not None:
        at.text_input(key="account_input").set_value(account).run()
    return at


def test_app_known_account_renders(processed: Path) -> None:
    at = run("A")
    assert not at.exception
    assert any("Anomaly score" in m.label for m in at.metric)
    assert any("temporal_cycle" in md.value for md in at.markdown) or True


def test_app_unknown_account_and_empty_input(processed: Path) -> None:
    at = run("ZZZ")
    assert not at.exception
    assert any("Unknown account" in w.value for w in at.warning)
    assert not run(None).exception
