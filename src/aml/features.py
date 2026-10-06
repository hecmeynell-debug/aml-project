"""Account-level features (Phase 4) and, later, leakage-free transaction features (Phase 5).

Account features describe each account's whole observed behaviour. They are *unsupervised*
inputs: no label is read here. Accounts are those with at least one non-self-loop transaction;
self-loops (e.g. "Reinvestment") carry no flow between accounts and are only counted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aml.config import Config
from aml.detectors import DETECTOR_NAMES, pass_through_metrics

logger = logging.getLogger(__name__)

ACCOUNT_FEATURES_FILE = "account_features.parquet"

# Columns that are heavy-tailed non-negative quantities: log1p-transformed before scaling.
LOG_COLUMNS: list[str] = [
    "n_in", "n_out", "deg_in", "deg_out", "usd_in_sum", "usd_in_mean", "usd_in_max",
    "usd_out_sum", "usd_out_mean", "usd_out_max", "span_hours", "max_txns_24h", "n_self_loops",
    "ratio_out_in", "median_dwell_hours", "n_counterparties", "pagerank_scaled",
]


def _format_column(fmt: str) -> str:
    return "share_fmt_" + fmt.lower().replace(" ", "_")


def build_account_features(cfg: Config) -> pd.DataFrame:
    """Compute account-level features (without detector counts) and save them to parquet.

    Returns a frame indexed by ``acc``. Columns: degrees (distinct counterparties) and counts,
    inflow/outflow USD (sum, mean, max), payment-format shares, cross-currency and round-amount
    shares, activity span, peak transactions in any 24-hour window, pass-through ratio and dwell
    time, PageRank and clustering coefficient on the aggregated (non-self-loop) graph.
    """
    p = cfg.transactions_parquet.as_posix()
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE t AS
        SELECT txn_id, timestamp, src, dst, amt_usd, amt_paid, payment_format,
               (pay_ccy <> recv_ccy) AS xc, pay_ccy, recv_ccy
        FROM read_parquet('{p}') WHERE src <> dst
    """)
    formats = [r[0] for r in con.execute(
        "SELECT DISTINCT payment_format FROM t ORDER BY 1").fetchall()]
    fmt_sql = ", ".join(
        f"avg((payment_format = '{f}')::INT) AS {_format_column(f)}" for f in formats)

    logger.info("Computing flow, mix and burst features")
    con.execute("""
        CREATE TABLE leg AS
        SELECT src AS acc, timestamp AS ts, payment_format, xc, amt_paid FROM t
        UNION ALL SELECT dst, timestamp, payment_format, xc, amt_paid FROM t
    """)
    out_s = con.execute("""
        SELECT src AS acc, count(*) n_out, count(DISTINCT dst) deg_out, sum(amt_usd) usd_out_sum,
               avg(amt_usd) usd_out_mean, max(amt_usd) usd_out_max FROM t GROUP BY 1""").df()
    in_s = con.execute("""
        SELECT dst AS acc, count(*) n_in, count(DISTINCT src) deg_in, sum(amt_usd) usd_in_sum,
               avg(amt_usd) usd_in_mean, max(amt_usd) usd_in_max FROM t GROUP BY 1""").df()
    mix = con.execute(f"""
        SELECT acc, {fmt_sql}, avg(xc::INT) AS cross_ccy_share,
               avg((amt_paid % 100 = 0)::INT) AS round_share,
               (epoch(max(ts)) - epoch(min(ts))) / 3600.0 AS span_hours
        FROM leg GROUP BY acc""").df()
    burst = con.execute("""
        SELECT acc, max(c) AS max_txns_24h FROM (
            SELECT acc, count(*) OVER (PARTITION BY acc ORDER BY ts
                   RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND CURRENT ROW) AS c FROM leg)
        GROUP BY acc""").df()
    self_loops = duckdb.sql(f"""
        SELECT src AS acc, count(*) AS n_self_loops FROM read_parquet('{p}')
        WHERE src = dst GROUP BY 1""").df()
    pairs = con.execute("SELECT src, dst FROM t GROUP BY 1, 2").df()

    feats = mix.merge(burst, on="acc", how="left")
    for part in (out_s, in_s, self_loops):
        feats = feats.merge(part, on="acc", how="left")
    count_cols = ["n_out", "deg_out", "usd_out_sum", "usd_out_mean", "usd_out_max",
                  "n_in", "deg_in", "usd_in_sum", "usd_in_mean", "usd_in_max", "n_self_loops"]
    feats[count_cols] = feats[count_cols].fillna(0)

    logger.info("Computing pass-through metrics")
    pt = pass_through_metrics(pd.read_parquet(
        cfg.transactions_parquet,
        columns=["txn_id", "timestamp", "src", "dst", "amt_usd", "pay_ccy", "recv_ccy"]))
    feats = feats.merge(
        pt[["acc", "ratio_out_in", "median_dwell_hours", "n_counterparties"]], on="acc", how="left")
    feats["has_both_directions"] = feats["ratio_out_in"].notna().astype(int)
    # No in+out pair -> not a conduit: zero ratio and the longest possible dwell.
    feats["ratio_out_in"] = feats["ratio_out_in"].fillna(0.0)
    feats["median_dwell_hours"] = feats["median_dwell_hours"].fillna(feats["span_hours"].max())
    feats["n_counterparties"] = feats["n_counterparties"].fillna(
        feats[["deg_in", "deg_out"]].max(axis=1))

    logger.info("Computing PageRank and clustering on %d account pairs", len(pairs))
    dg = nx.from_pandas_edgelist(pairs, "src", "dst", create_using=nx.DiGraph)
    pr = pd.Series(nx.pagerank(dg, alpha=0.85, max_iter=100, tol=1e-8), name="pagerank")
    clust = pd.Series(nx.clustering(dg.to_undirected()), name="clustering")
    feats = feats.merge(pr.rename_axis("acc").reset_index(), on="acc", how="left")
    feats = feats.merge(clust.rename_axis("acc").reset_index(), on="acc", how="left")
    feats["pagerank_scaled"] = feats["pagerank"] * len(pr)  # 1.0 = average node
    feats = feats.drop(columns="pagerank").set_index("acc").sort_index()

    path = cfg.root / cfg.paths.processed_dir / ACCOUNT_FEATURES_FILE
    feats.to_parquet(path)
    logger.info("Saved %d accounts x %d features to %s", *feats.shape, path)
    return feats


def add_detector_counts(feats: pd.DataFrame, findings: pd.DataFrame) -> pd.DataFrame:
    """Add ``hits_<detector>``: how many of that detector's findings include each account."""
    out = feats.drop(columns=[c for c in feats.columns if c.startswith("hits_")])
    hits = (
        findings[["detector", "account_ids"]].explode("account_ids")
        .rename(columns={"account_ids": "acc"}).dropna()
        .groupby(["acc", "detector"]).size().unstack(fill_value=0)
    )
    hits = hits.reindex(columns=DETECTOR_NAMES, fill_value=0)
    hits.columns = [f"hits_{c}" for c in hits.columns]
    out = out.join(hits, how="left")
    out[hits.columns] = out[hits.columns].fillna(0).astype(int)
    return out


def transform_features(feats: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """log1p the heavy-tailed columns (and detector hit counts), then standardise everything."""
    x = feats.copy()
    x["ratio_out_in"] = x["ratio_out_in"].clip(upper=100.0)  # tiny inflows give absurd ratios
    for col in [*LOG_COLUMNS, *[c for c in x.columns if c.startswith("hits_")]]:
        if col in x:
            x[col] = np.log1p(x[col].clip(lower=0))
    x = x.fillna(0.0)
    scaled = StandardScaler().fit_transform(x.to_numpy(dtype="float64"))
    return scaled, list(x.columns)


# --------------------------------------------------------------------------- Phase 5: transactions
TXN_FEATURES_FILE = "txn_features.parquet"
CATEGORICAL_FEATURES = ["payment_format", "pay_ccy", "recv_ccy"]


def build_transaction_features(cfg: Config) -> Path:
    """Leakage-free transaction-level features, written to ``txn_features.parquet``.

    Every history feature for a transaction at time ``ts`` uses only transactions with timestamp
    **strictly earlier** than ``ts`` (same-minute transactions are excluded, which also excludes
    the transaction itself). For lookback windows ``w`` in ``supervised.lookback_days`` the
    window is ``[ts - w, ts)``.

    For the *sender* the features describe its outgoing history (count, USD volume, distinct
    receivers, cross-currency share) plus its incoming count and USD volume; for the *receiver*
    they describe its incoming history (count, USD volume, distinct senders, cross-currency share)
    plus its outgoing count and USD volume. Counts and volumes come from cumulative sums looked
    up with ASOF joins; distinct counts use DuckDB window functions with a RANGE frame.

    Own-transaction features: ``log_amt_usd``, payment format, currencies, cross-currency flag,
    hour, weekday, round-amount flag, self-loop flag and ``pair_seen_before``. The label is
    carried through as ``is_laundering`` for training and evaluation only.
    """
    windows = tuple(cfg.supervised.lookback_days)
    p = cfg.transactions_parquet.as_posix()
    out = cfg.root / cfg.paths.processed_dir / TXN_FEATURES_FILE
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='" + (cfg.root / "data" / "processed").as_posix() + "'")
    con.execute(f"""
        CREATE TABLE base AS
        SELECT txn_id, timestamp AS ts, src, dst, amt_usd, amt_paid, payment_format, pay_ccy,
               recv_ccy, is_laundering, (src = dst) AS is_self_loop,
               (pay_ccy <> recv_ccy)::INT AS xc
        FROM read_parquet('{p}')
    """)
    # cumulative activity per (account, role) at each distinct timestamp
    for role, key in (("out", "src"), ("in", "dst")):
        con.execute(f"""
            CREATE TABLE cum_{role} AS
            SELECT {key} AS acc, ts,
                   sum(c) OVER w AS cnt, sum(u) OVER w AS usd, sum(x) OVER w AS xc
            FROM (SELECT {key}, ts, count(*) AS c, sum(amt_usd) AS u, sum(xc) AS x
                  FROM base GROUP BY 1, 2)
            WINDOW w AS (PARTITION BY {key} ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        """)
    lookups = [("0", "b.ts")] + [(str(w), f"b.ts - INTERVAL {w} DAY") for w in windows]
    selects, joins = [], []
    n = 0
    # (party column, role table, feature prefix)
    for party, role, prefix in (("src", "out", "snd_out"), ("src", "in", "snd_in"),
                                ("dst", "in", "rcv_in"), ("dst", "out", "rcv_out")):
        for tag, expr in lookups:
            a = f"j{n}"; n += 1
            joins.append(
                f"ASOF LEFT JOIN cum_{role} {a} ON b.{party} = {a}.acc AND ({expr}) > {a}.ts")
            for m in ("cnt", "usd", "xc"):
                selects.append(f"COALESCE({a}.{m}, 0) AS {prefix}_{m}_L{tag}")
    con.execute("CREATE TABLE lk AS SELECT b.txn_id, " + ", ".join(selects) +
                " FROM base b " + " ".join(joins))
    logger.info("History lookups done")

    # distinct counterparties via RANGE window frames: [ts - w, ts - 1 second]
    dcols = []
    for side, key, other in (("snd_out", "src", "dst"), ("rcv_in", "dst", "src")):
        parts = ", ".join(
            f"count(DISTINCT {other}) OVER (PARTITION BY {key} ORDER BY ts "
            f"RANGE BETWEEN INTERVAL {w} DAY PRECEDING AND INTERVAL 1 SECOND PRECEDING) "
            f"AS {side}_dcp_{w}d" for w in windows)
        con.execute(f"CREATE TABLE d_{side} AS SELECT txn_id, {parts} FROM base")
        dcols.append(side)
        logger.info("Distinct counterparties for %s done", side)

    pair = ("SELECT b.txn_id, (pf.first_ts < b.ts)::INT AS pair_seen_before FROM base b "
            "JOIN (SELECT src, dst, min(ts) AS first_ts FROM base GROUP BY 1, 2) pf "
            "USING (src, dst)")
    con.execute(f"CREATE TABLE pr AS {pair}")

    # windowed = cumulative(ts) - cumulative(ts - w)
    wcols = []
    for prefix in ("snd_out", "snd_in", "rcv_in", "rcv_out"):
        for w in windows:
            wcols.append(f"lk.{prefix}_cnt_L0 - lk.{prefix}_cnt_L{w} AS {prefix}_cnt_{w}d")
            wcols.append(f"lk.{prefix}_usd_L0 - lk.{prefix}_usd_L{w} AS {prefix}_usd_{w}d")
            if prefix in ("snd_out", "rcv_in"):
                wcols.append(
                    f"CASE WHEN lk.{prefix}_cnt_L0 - lk.{prefix}_cnt_L{w} > 0 THEN "
                    f"(lk.{prefix}_xc_L0 - lk.{prefix}_xc_L{w}) / "
                    f"(lk.{prefix}_cnt_L0 - lk.{prefix}_cnt_L{w}) ELSE 0 END AS {prefix}_xc_share_{w}d")
    dsel = ", ".join(
        f"d_snd_out.snd_out_dcp_{w}d, d_rcv_in.rcv_in_dcp_{w}d" for w in windows)
    con.execute(f"""
        COPY (
            SELECT b.txn_id, b.ts AS timestamp, b.src, b.dst, b.is_laundering,
                   LN(1 + b.amt_usd) AS log_amt_usd, b.payment_format, b.pay_ccy, b.recv_ccy,
                   b.xc AS is_cross_ccy, hour(b.ts) AS hour, dayofweek(b.ts) AS weekday,
                   (b.amt_paid % 100 = 0)::INT AS is_round_amt, b.is_self_loop::INT AS is_self_loop,
                   pr.pair_seen_before, {", ".join(wcols)}, {dsel}
            FROM base b JOIN lk USING (txn_id) JOIN pr USING (txn_id)
                 JOIN d_snd_out USING (txn_id) JOIN d_rcv_in USING (txn_id)
            ORDER BY b.ts, b.txn_id
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    logger.info("Saved transaction features to %s", out)
    return out
