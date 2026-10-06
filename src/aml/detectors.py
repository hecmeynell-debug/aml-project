"""Rule-based typology detectors.

Every detector takes the transactions frame (needs ``txn_id, timestamp, src, dst, amt_usd``, and
``pay_ccy, recv_ccy`` for pass-through) and a :class:`~aml.config.DetectorsConfig`, and returns a
tidy findings frame with columns :data:`FINDING_COLUMNS`. The ``is_laundering`` label is never
read here; it is used only by :mod:`aml.evaluate`.

Self-loops are ignored throughout: they carry no flow between accounts.

Pass-through note: the dataset has no jurisdiction or "offshore" field, so shell-like accounts
cannot be identified directly. :func:`pass_through` flags accounts that behave like a conduit
(outflow ~ inflow, short dwell). This is a **behavioural proxy** for shell accounts, not a
shell-company identification.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from aml.config import Config, DetectorsConfig
from aml.graph import component_labels

logger = logging.getLogger(__name__)

FINDING_COLUMNS: list[str] = [
    "detector", "account_ids", "txn_ids", "start_ts", "end_ts", "total_usd", "score", "detail",
]
DETECTOR_NAMES: list[str] = [
    "temporal_cycle", "fan_in", "fan_out", "scatter_gather", "gather_scatter", "pass_through",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FINDING_COLUMNS})


def _frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return _empty()
    out = pd.DataFrame(rows, columns=FINDING_COLUMNS)
    out["start_ts"] = pd.to_datetime(out["start_ts"])
    out["end_ts"] = pd.to_datetime(out["end_ts"])
    return out


def _con(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with the non-self-loop transactions registered as table ``t``."""
    cols = ["txn_id", "timestamp", "src", "dst", "amt_usd", *extra]
    con = duckdb.connect()
    con.register("t_raw", df.loc[df["src"] != df["dst"], cols])
    con.execute("CREATE TABLE t AS SELECT * FROM t_raw")
    return con


# --------------------------------------------------------------------------- temporal cycles
def temporal_cycles(df: pd.DataFrame, cfg: DetectorsConfig) -> pd.DataFrame:
    """Cycles realised by a strictly time-ordered chain of transactions.

    A finding is a loop ``a0 -> a1 -> ... -> a0`` of at most ``cycle_length_bound`` hops such that
    every hop is strictly later than the previous one, the whole loop fits within
    ``cycle_window_days`` of the first hop, and every hop's USD amount is within
    ``amount_tolerance`` of the first hop's.

    The search is restricted to non-trivial strongly connected components of the account graph
    (the only place a cycle can exist). Inside a component it is a time-respecting depth-first
    search that starts at each transaction and only follows later, amount-compatible
    transactions. That returns exactly the cycles an "enumerate simple cycles, then verify the
    timing" pipeline would accept, without enumerating the (enormous) number of cycles that
    fail the time test. Each cycle is found once, from its earliest hop.

    Safety caps (``max_cycles_per_component`` found cycles, ``max_expansions_per_component``
    edge examinations) are logged when hit, in which case results for that component are partial.
    """
    e = df.loc[df["src"] != df["dst"], ["txn_id", "timestamp", "src", "dst", "amt_usd"]]
    if e.empty:
        return _empty()
    codes, nodes = pd.factorize(pd.concat([e["src"], e["dst"]], ignore_index=True))
    n = len(e)
    s_code, d_code = codes[:n], codes[n:]
    scc, _ = component_labels(s_code, d_code, len(nodes))
    sizes = np.bincount(scc)
    keep = (scc[s_code] == scc[d_code]) & (sizes[scc[s_code]] > 1)
    sub = e[keep].assign(s=s_code[keep], d=d_code[keep], comp=scc[s_code][keep])
    sub = sub.assign(sec=sub["timestamp"].astype("datetime64[s]").astype("int64"))
    logger.info(
        "temporal_cycles: %d of %d transactions lie in %d non-trivial SCCs",
        len(sub), n, sub["comp"].nunique(),
    )

    window = cfg.cycle_window_days * 86400
    tol, max_len, min_len = cfg.amount_tolerance, cfg.cycle_length_bound, cfg.cycle_min_length
    rows: list[dict] = []
    n_capped = 0
    for comp, g in sub.groupby("comp", sort=False):
        adj: dict[int, tuple[list, list, list, list]] = {}
        g = g.sort_values(["sec", "txn_id"])
        grouped: dict[int, list] = defaultdict(lambda: [[], [], [], []])
        for sec, amt, d, tid, s in zip(g["sec"], g["amt_usd"], g["d"], g["txn_id"], g["s"]):
            lst = grouped[s]
            lst[0].append(sec); lst[1].append(amt); lst[2].append(d); lst[3].append(tid)
        adj = {k: tuple(v) for k, v in grouped.items()}  # type: ignore[assignment]
        info = g.set_index("txn_id")[["timestamp", "amt_usd"]]
        found: list[tuple[list[int], list[int]]] = []
        state = {"expansions": 0, "capped": False}

        def extend(start: int, node: int, last_t: int, t_end: int, lo: float, hi: float,
                   path_nodes: list[int], path_edges: list[int]) -> None:
            tl, al, dl, il = adj.get(node, ((), (), (), ()))
            i = bisect_right(tl, last_t)
            while i < len(tl) and tl[i] <= t_end:
                if state["expansions"] >= cfg.max_expansions_per_component:
                    state["capped"] = True
                    return
                state["expansions"] += 1
                if lo <= al[i] <= hi:
                    w = dl[i]
                    if w == start:
                        if len(path_edges) + 1 >= min_len:
                            found.append((list(path_nodes), [*path_edges, il[i]]))
                            if len(found) >= cfg.max_cycles_per_component:
                                state["capped"] = True
                                return
                    elif len(path_edges) + 1 < max_len and w not in path_nodes:
                        path_nodes.append(w); path_edges.append(il[i])
                        extend(start, w, tl[i], t_end, lo, hi, path_nodes, path_edges)
                        path_nodes.pop(); path_edges.pop()
                        if state["capped"]:
                            return
                i += 1

        for u, (tl, al, dl, il) in adj.items():
            for t0, a0, v, id0 in zip(tl, al, dl, il):
                extend(u, v, t0, t0 + window, a0 * (1 - tol), a0 * (1 + tol), [u, v], [id0])
                if state["capped"]:
                    break
            if state["capped"]:
                break
        if state["capped"]:
            n_capped += 1
            logger.warning(
                "temporal_cycles: cap hit in SCC %s (%d nodes, %d cycles found, %d expansions)",
                comp, len(adj), len(found), state["expansions"],
            )
        for path_nodes, edges in found:
            sel = info.loc[edges]
            first = sel["amt_usd"].iloc[0]
            dev = float((sel["amt_usd"] / first - 1).abs().max())
            rows.append({
                "detector": "temporal_cycle",
                "account_ids": [nodes[k] for k in path_nodes],
                "txn_ids": [int(x) for x in edges],
                "start_ts": sel["timestamp"].min(), "end_ts": sel["timestamp"].max(),
                "total_usd": float(sel["amt_usd"].sum()),
                "score": 0.5 + 0.5 * (1 - min(dev / tol, 1.0)),
                "detail": f"{len(edges)} hops, max amount deviation {dev:.1%}",
            })
    if n_capped:
        logger.warning("temporal_cycles: %d component(s) hit a safety cap", n_capped)
    logger.info("temporal_cycles: %d cycles", len(rows))
    return _frame(rows)


# --------------------------------------------------------------------------- fan-in / fan-out
def _fan(df: pd.DataFrame, cfg: DetectorsConfig, direction: str) -> pd.DataFrame:
    """Shared implementation of fan-out (``direction='out'``) and fan-in (``'in'``).

    For each account and window length, the peak number of distinct counterparties inside any
    sliding window ending at one of its transactions is computed with DuckDB window functions.
    An account is flagged when its peak is strictly above the ``fan_percentile`` of the same
    statistic over all accounts (and at least ``fan_min_degree``). Accounts with fewer than
    ``fan_min_degree`` counterparties in total cannot be flagged and enter the population at
    their total distinct count (an upper bound on their peak).
    """
    key, other = ("src", "dst") if direction == "out" else ("dst", "src")
    name = f"fan_{direction}"
    con = _con(df)
    con.execute(f"""
        CREATE TABLE deg AS SELECT {key} AS k, count(DISTINCT {other}) AS nd FROM t GROUP BY 1
    """)
    con.execute(f"CREATE TABLE cand AS SELECT k FROM deg WHERE nd >= {cfg.fan_min_degree}")
    small = con.execute(f"SELECT nd FROM deg WHERE nd < {cfg.fan_min_degree}").df()["nd"].to_numpy()

    best: dict[str, dict] = {}
    for w in cfg.fan_windows_days:
        peaks = con.execute(f"""
            WITH win AS (
                SELECT {key} AS k, timestamp AS ts,
                       count(DISTINCT {other}) OVER (
                           PARTITION BY {key} ORDER BY timestamp
                           RANGE BETWEEN INTERVAL {int(w)} DAYS PRECEDING AND CURRENT ROW) AS c
                FROM t WHERE {key} IN (SELECT k FROM cand))
            SELECT k, max(c) AS peak, arg_max(ts, c) AS peak_ts FROM win GROUP BY k
        """).df()
        population = np.concatenate([peaks["peak"].to_numpy(), small])
        thr = max(float(np.percentile(population, cfg.fan_percentile)), cfg.fan_min_degree - 1)
        flagged = peaks[peaks["peak"] > thr]
        logger.info("%s w=%dd: threshold %.1f, %d accounts flagged", name, w, thr, len(flagged))
        for r in flagged.itertuples():
            score = 1 - thr / r.peak
            if r.k not in best or score > best[r.k]["score"]:
                best[r.k] = {"score": score, "w": w, "peak": int(r.peak), "peak_ts": r.peak_ts,
                             "thr": thr}
    if not best:
        return _empty()

    sel = pd.DataFrame([{"k": k, **v} for k, v in best.items()])
    con.register("sel", sel)
    cap = cfg.max_txns_per_finding
    res = con.execute(f"""
        SELECT s.k, s.w, s.peak, s.score, s.thr,
               list(t.txn_id ORDER BY t.amt_usd DESC)[1:{cap}] AS txn_ids,
               list(DISTINCT t.{other}) AS others,
               min(t.timestamp) AS start_ts, max(t.timestamp) AS end_ts, sum(t.amt_usd) AS total
        FROM sel s JOIN t ON t.{key} = s.k
         AND t.timestamp <= s.peak_ts AND t.timestamp >= s.peak_ts - s.w * INTERVAL 1 DAY
        GROUP BY s.k, s.w, s.peak, s.score, s.thr
    """).df()
    rows = [{
        "detector": name,
        "account_ids": [r.k, *list(r.others)[: cap]],
        "txn_ids": [int(x) for x in r.txn_ids],
        "start_ts": r.start_ts, "end_ts": r.end_ts, "total_usd": float(r.total),
        "score": float(r.score),
        "detail": f"{r.peak} distinct counterparties in {r.w}d (threshold {r.thr:.0f})",
    } for r in res.itertuples()]
    return _frame(rows)


def fan_out(df: pd.DataFrame, cfg: DetectorsConfig) -> pd.DataFrame:
    """Accounts paying an unusually large number of distinct counterparties in a short window."""
    return _fan(df, cfg, "out")


def fan_in(df: pd.DataFrame, cfg: DetectorsConfig) -> pd.DataFrame:
    """Accounts receiving from an unusually large number of distinct counterparties."""
    return _fan(df, cfg, "in")


# --------------------------------------------------------------------------- scatter-gather
def _two_hop_paths(con: duckdb.DuckDBPyConnection, cfg: DetectorsConfig) -> int:
    """Materialise time-ordered two-hop paths S -> I -> D (table ``paths``); return their count.

    Intermediaries with more than ``max_intermediary_txns`` incoming or outgoing transactions
    are skipped (they are hubs, and their path count grows quadratically).
    """
    cap = cfg.max_intermediary_txns
    w = cfg.scatter_gather_window_days
    con.execute(f"""
        CREATE OR REPLACE TABLE ok_int AS
        SELECT i.acc FROM (SELECT dst AS acc, count(*) c FROM t GROUP BY 1) i
        JOIN (SELECT src AS acc, count(*) c FROM t GROUP BY 1) o USING (acc)
        WHERE i.c <= {cap} AND o.c <= {cap}
    """)
    skipped = con.execute("""
        SELECT count(*) FROM (SELECT dst AS acc FROM t INTERSECT SELECT src FROM t)
        WHERE acc NOT IN (SELECT acc FROM ok_int)""").fetchone()[0]
    if skipped:
        logger.info("two-hop paths: skipped %d hub intermediaries", skipped)
    con.execute(f"""
        CREATE OR REPLACE TABLE paths AS
        SELECT a.src AS s, a.dst AS i, b.dst AS d, a.txn_id AS t1, b.txn_id AS t2,
               a.timestamp AS ts1, b.timestamp AS ts2, a.amt_usd AS u1, b.amt_usd AS u2
        FROM t a JOIN t b ON a.dst = b.src
        WHERE a.dst IN (SELECT acc FROM ok_int) AND b.timestamp > a.timestamp
          AND b.timestamp <= a.timestamp + {float(w)} * INTERVAL 1 DAY AND a.src <> b.dst
    """)
    n = con.execute("SELECT count(*) FROM paths").fetchone()[0]
    logger.info("two-hop paths: %d", n)
    return n


def scatter_gather(df: pd.DataFrame, cfg: DetectorsConfig, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Source/destination pairs joined by >= ``scatter_gather_min_paths`` time-ordered two-hop paths
    through distinct intermediaries within ``scatter_gather_window_days``."""
    con = con or _con(df)
    if "paths" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
        _two_hop_paths(con, cfg)
    m, cap = cfg.scatter_gather_min_paths, cfg.max_txns_per_finding
    res = con.execute(f"""
        SELECT s, d, count(DISTINCT i) AS n_int, list(DISTINCT i) AS ints,
               list(DISTINCT t1) AS a, list(DISTINCT t2) AS b,
               min(ts1) AS start_ts, max(ts2) AS end_ts, sum(u1) AS total
        FROM paths GROUP BY s, d HAVING count(DISTINCT i) >= {m}
    """).df()
    rows = [{
        "detector": "scatter_gather",
        "account_ids": [r.s, *list(r.ints), r.d],
        "txn_ids": [int(x) for x in (list(r.a) + list(r.b))][:cap],
        "start_ts": r.start_ts, "end_ts": r.end_ts, "total_usd": float(r.total),
        "score": min(1.0, r.n_int / (2 * m)),
        "detail": f"{r.n_int} intermediaries",
    } for r in res.itertuples()]
    logger.info("scatter_gather: %d findings", len(rows))
    return _frame(rows)


def gather_scatter(df: pd.DataFrame, cfg: DetectorsConfig, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Hub accounts that collect from >= ``min_paths`` distinct sources and then redistribute to
    >= ``min_paths`` distinct destinations within the window (each outgoing payment after an
    incoming one).

    Note: the brief defines scatter-gather and gather-scatter by the same source/destination
    pairing; for gather-scatter the natural unit is the *hub*, so it is detected per hub.
    """
    con = con or _con(df)
    if "paths" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
        _two_hop_paths(con, cfg)
    m, cap = cfg.scatter_gather_min_paths, cfg.max_txns_per_finding
    res = con.execute(f"""
        SELECT i, count(DISTINCT s) AS n_src, count(DISTINCT d) AS n_dst,
               list(DISTINCT s) AS srcs, list(DISTINCT d) AS dsts,
               list(DISTINCT t1) AS a, list(DISTINCT t2) AS b,
               min(ts1) AS start_ts, max(ts2) AS end_ts, sum(u1) AS total
        FROM paths GROUP BY i HAVING count(DISTINCT s) >= {m} AND count(DISTINCT d) >= {m}
    """).df()
    rows = [{
        "detector": "gather_scatter",
        "account_ids": [r.i, *list(r.srcs)[:cap], *list(r.dsts)[:cap]],
        "txn_ids": [int(x) for x in (list(r.a) + list(r.b))][:cap],
        "start_ts": r.start_ts, "end_ts": r.end_ts, "total_usd": float(r.total),
        "score": min(1.0, min(r.n_src, r.n_dst) / (2 * m)),
        "detail": f"{r.n_src} sources -> hub -> {r.n_dst} destinations",
    } for r in res.itertuples()]
    logger.info("gather_scatter: %d findings", len(rows))
    return _frame(rows)


# --------------------------------------------------------------------------- pass-through
def pass_through_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-account conduit metrics (reused by :mod:`aml.features`).

    Columns: ``acc, n_in, n_out, usd_in, usd_out, ratio_out_in, median_dwell_hours,
    n_counterparties, cross_ccy_share``. Dwell time is, for each incoming transaction, the time
    to the account's next outgoing transaction (same or later timestamp); the median is taken
    over incoming transactions that have one. Accounts need at least one incoming and one
    outgoing transaction.
    """
    con = _con(df, extra=("pay_ccy", "recv_ccy"))
    con.execute("""
        CREATE TABLE inn AS SELECT dst AS acc, timestamp AS ts FROM t;
        CREATE TABLE outt AS SELECT src AS acc, timestamp AS ts FROM t;
    """)
    out = con.execute("""
        WITH i AS (SELECT dst AS acc, count(*) n_in, sum(amt_usd) usd_in FROM t GROUP BY 1),
             o AS (SELECT src AS acc, count(*) n_out, sum(amt_usd) usd_out FROM t GROUP BY 1),
             dw AS (SELECT inn.acc, median(epoch(outt.ts) - epoch(inn.ts)) / 3600.0 AS dwell_h
                    FROM inn ASOF JOIN outt ON inn.acc = outt.acc AND inn.ts <= outt.ts
                    GROUP BY inn.acc),
             cp AS (SELECT acc, count(DISTINCT other) AS n_cp FROM (
                        SELECT src AS acc, dst AS other FROM t
                        UNION ALL SELECT dst, src FROM t) GROUP BY acc),
             xc AS (SELECT acc, avg(x) AS share FROM (
                        SELECT src AS acc, (pay_ccy <> recv_ccy)::INT AS x FROM t
                        UNION ALL SELECT dst, (pay_ccy <> recv_ccy)::INT FROM t) GROUP BY acc)
        SELECT i.acc, n_in, n_out, usd_in, usd_out, usd_out / usd_in AS ratio_out_in,
               dw.dwell_h AS median_dwell_hours, cp.n_cp AS n_counterparties,
               xc.share AS cross_ccy_share
        FROM i JOIN o USING (acc) LEFT JOIN dw USING (acc) LEFT JOIN cp USING (acc)
        LEFT JOIN xc USING (acc)
    """).df()
    return out


def pass_through(df: pd.DataFrame, cfg: DetectorsConfig) -> pd.DataFrame:
    """Accounts that behave like a conduit: outflow ~ inflow and a short dwell time.

    Flagged when: at least ``pass_through_min_txns`` incoming and outgoing transactions, total
    inflow >= ``pass_through_min_usd``, ``|outflow / inflow - 1| <= pass_through_ratio_tol`` (low
    retention) and median dwell <= ``pass_through_max_dwell_hours``.

    This is a behavioural proxy for shell accounts; see the module docstring.
    """
    m = pass_through_metrics(df)
    ok = m[
        (m["n_in"] >= cfg.pass_through_min_txns) & (m["n_out"] >= cfg.pass_through_min_txns)
        & (m["usd_in"] >= cfg.pass_through_min_usd)
        & ((m["ratio_out_in"] - 1).abs() <= cfg.pass_through_ratio_tol)
        & (m["median_dwell_hours"] <= cfg.pass_through_max_dwell_hours)
    ].copy()
    logger.info("pass_through: %d of %d accounts flagged", len(ok), len(m))
    if ok.empty:
        return _empty()
    con = _con(df)
    con.register("flag", ok[["acc"]])
    tx = con.execute("""
        SELECT f.acc, list(txn_id ORDER BY amt_usd DESC) AS ids,
               min(timestamp) AS s, max(timestamp) AS e
        FROM flag f JOIN t ON t.src = f.acc OR t.dst = f.acc GROUP BY f.acc
    """).df().set_index("acc")
    cap = cfg.max_txns_per_finding
    rows = []
    for r in ok.itertuples():
        t = tx.loc[r.acc]
        dev = abs(r.ratio_out_in - 1) / cfg.pass_through_ratio_tol
        dwell = r.median_dwell_hours / cfg.pass_through_max_dwell_hours
        rows.append({
            "detector": "pass_through", "account_ids": [r.acc],
            "txn_ids": [int(x) for x in list(t["ids"])[:cap]],
            "start_ts": t["s"], "end_ts": t["e"], "total_usd": float(r.usd_in),
            "score": float(0.5 * (1 - min(dev, 1)) + 0.5 * (1 - min(dwell, 1))),
            "detail": f"out/in {r.ratio_out_in:.2f}, median dwell {r.median_dwell_hours:.1f}h",
        })
    return _frame(rows)


# --------------------------------------------------------------------------- orchestration
def run_detectors(df: pd.DataFrame, cfg: DetectorsConfig) -> pd.DataFrame:
    """Run every detector and concatenate the findings."""
    con = _con(df)
    parts = [
        temporal_cycles(df, cfg),
        fan_out(df, cfg),
        fan_in(df, cfg),
        scatter_gather(df, cfg, con),
        gather_scatter(df, cfg, con),
        pass_through(df.assign(), cfg) if {"pay_ccy", "recv_ccy"} <= set(df.columns) else _empty(),
    ]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for name, grp in out.groupby("detector"):
        logger.info("%-15s %7d findings, %9d txns covered", name, len(grp),
                    len({t for ids in grp["txn_ids"] for t in ids}))
    return out


def detect_all(cfg: Config) -> pd.DataFrame:
    """Load the processed transactions, run all detectors and save ``findings.parquet``."""
    cols = ["txn_id", "timestamp", "src", "dst", "amt_usd", "pay_ccy", "recv_ccy"]
    df = pd.read_parquet(cfg.transactions_parquet, columns=cols)
    findings = run_detectors(df, cfg.detectors)
    path = cfg.root / cfg.paths.processed_dir / "findings.parquet"
    findings.to_parquet(path, index=False)
    logger.info("Saved %d findings to %s", len(findings), path)
    return findings
