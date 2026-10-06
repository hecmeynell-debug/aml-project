"""Precompute the score tables the dashboard reads (it never trains or scores on the fly).

Outputs (all in ``data/processed/``):

* ``transaction_scores.parquet``: every transaction with its supervised score and train/val/test
  split. Scores on the *train* and *validation* splits are in-sample for the model and optimistic;
  only the ``test`` split is a fair evaluation.
* ``account_scores.parquet``: ensemble anomaly score, maximum supervised score over the account's
  transactions, detector hit counts, USD flows and the top three human-readable reasons.
* ``findings.parquet``: every detector finding (written by :mod:`aml.detectors`).
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from aml.config import Config
from aml.features import TXN_FEATURES_FILE, add_detector_counts, transform_features
from aml.supervised import load_model, prepare_features

logger = logging.getLogger(__name__)

TXN_SCORES_FILE = "transaction_scores.parquet"
ACCOUNT_SCORES_FILE = "account_scores.parquet"

FRIENDLY = {
    "n_in": "incoming transaction count", "n_out": "outgoing transaction count",
    "deg_in": "number of distinct senders", "deg_out": "number of distinct recipients",
    "usd_in_sum": "total USD received", "usd_in_mean": "mean USD received",
    "usd_in_max": "largest USD received", "usd_out_sum": "total USD sent",
    "usd_out_mean": "mean USD sent", "usd_out_max": "largest USD sent",
    "span_hours": "activity span", "max_txns_24h": "peak transactions in 24h",
    "n_self_loops": "self-loop count", "ratio_out_in": "outflow/inflow ratio",
    "median_dwell_hours": "median dwell time", "n_counterparties": "number of counterparties",
    "pagerank_scaled": "PageRank", "clustering": "clustering coefficient",
    "cross_ccy_share": "cross-currency share", "round_share": "round-amount share",
}
DETECTOR_TEXT = {
    "temporal_cycle": "in a time-ordered cycle", "fan_out": "fan-out burst",
    "fan_in": "fan-in burst", "scatter_gather": "scatter-gather structure",
    "gather_scatter": "gather-scatter hub", "pass_through": "pass-through behaviour",
}


def score_transactions(cfg: Config) -> pd.DataFrame:
    """Score every transaction with the saved model (streamed in batches) and attach metadata."""
    proc = cfg.root / cfg.paths.processed_dir
    model, meta = load_model(cfg)
    pf = pq.ParquetFile(proc / TXN_FEATURES_FILE)
    parts = []
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        x = prepare_features(df, meta["categories"])[meta["features"]]
        parts.append(pd.DataFrame({
            "txn_id": df["txn_id"].to_numpy(),
            "score": model.predict(x, num_iteration=meta["best_iteration"]),
        }))
    scores = pd.concat(parts, ignore_index=True)
    scores["flagged"] = scores["score"] >= meta["threshold"]
    split = pd.read_parquet(proc / "txn_split.parquet")
    scores = scores.merge(split, on="txn_id", how="left")
    con = duckdb.connect()
    con.register("sc", scores)
    out = con.execute(f"""
        SELECT t.txn_id, t.timestamp, t.src, t.dst, t.amt_usd, t.payment_format, t.pay_ccy,
               t.recv_ccy, t.is_laundering, sc.score, sc.flagged, sc.split
        FROM read_parquet('{cfg.transactions_parquet.as_posix()}') t JOIN sc USING (txn_id)
        ORDER BY t.txn_id
    """).df()
    out.to_parquet(proc / TXN_SCORES_FILE, index=False)
    logger.info("Wrote %d transaction scores", len(out))
    return out


def _top_feature_reasons(feats: pd.DataFrame, top: int = 3) -> list[list[str]]:
    """For each account, its most extreme standardised features as short sentences."""
    x, cols = transform_features(feats)
    keep = [i for i, c in enumerate(cols) if c in FRIENDLY]
    z = x[:, keep]
    names = [FRIENDLY[cols[i]] for i in keep]
    order = np.argsort(-np.abs(z), axis=1)[:, :top]
    reasons = []
    for r in range(len(z)):
        reasons.append([
            f"{names[j]} unusually {'high' if z[r, j] > 0 else 'low'} (z={z[r, j]:+.1f})"
            for j in order[r]
        ])
    return reasons


def build_account_scores(cfg: Config, txn_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Combine anomaly scores, supervised scores, detector hits and reasons per account."""
    proc = cfg.root / cfg.paths.processed_dir
    findings = pd.read_parquet(proc / "findings.parquet")
    feats = add_detector_counts(pd.read_parquet(proc / "account_features.parquet"), findings)
    anom = pd.read_parquet(proc / "anomaly_scores.parquet")
    if txn_scores is None:
        txn_scores = pd.read_parquet(proc / TXN_SCORES_FILE)

    con = duckdb.connect()
    con.register("ts", txn_scores[["src", "dst", "score", "is_laundering"]])
    sup = con.execute("""
        SELECT acc, max(score) AS max_supervised_score, sum(is_laundering) AS n_laundering_txns
        FROM (SELECT src AS acc, score, is_laundering FROM ts
              UNION ALL SELECT dst, score, is_laundering FROM ts) GROUP BY acc
    """).df().set_index("acc")

    acc = feats.join(anom[["iforest", "lof", "hdbscan", "ensemble"]], how="left")
    acc = acc.rename(columns={"ensemble": "anomaly_score"})
    acc = acc.join(sup, how="left")
    acc["max_supervised_score"] = acc["max_supervised_score"].fillna(0.0)
    acc["is_illicit_account"] = (acc["n_laundering_txns"].fillna(0) > 0).astype(int)
    # Single triage number: mean of the two normalised ranks.
    r1 = rankdata(acc["anomaly_score"].fillna(0)) / len(acc)
    r2 = rankdata(acc["max_supervised_score"]) / len(acc)
    acc["risk"] = (r1 + r2) / 2

    hit_cols = [c for c in acc.columns if c.startswith("hits_")]
    feat_reasons = _top_feature_reasons(feats.loc[acc.index].drop(
        columns=[c for c in feats.columns if c.startswith("hits_")]))
    reasons = []
    for i, (row_hits, fr) in enumerate(zip(acc[hit_cols].to_numpy(), feat_reasons)):
        det = [f"{DETECTOR_TEXT[c.removeprefix('hits_')]} (x{int(h)})"
               for c, h in zip(hit_cols, row_hits) if h > 0]
        reasons.append((det + fr)[:3])
    acc["reasons"] = reasons
    acc["usd_in"] = acc["usd_in_sum"]
    acc["usd_out"] = acc["usd_out_sum"]
    keep = ["anomaly_score", "iforest", "lof", "hdbscan", "max_supervised_score", "risk",
            *hit_cols, "usd_in", "usd_out", "n_in", "n_out", "n_laundering_txns",
            "is_illicit_account", "reasons"]
    out = acc[keep].rename_axis("acc")
    out.to_parquet(proc / ACCOUNT_SCORES_FILE)
    logger.info("Wrote %d account scores", len(out))
    return out


def build_all_scores(cfg: Config) -> None:
    """Produce the three files the dashboard needs (``findings.parquet`` already exists)."""
    ts = score_transactions(cfg)
    build_account_scores(cfg, ts)
