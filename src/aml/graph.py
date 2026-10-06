"""Graph construction, connected components and DuckDB-backed ego-network extraction.

Scale note: the full HI-Small graph has ~5M transactions and ~0.5M accounts, so NetworkX is used
only for subgraphs. Whole-graph structure (degrees, components) goes through DuckDB and
``scipy.sparse.csgraph`` instead.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

from aml.config import Config, load_config

logger = logging.getLogger(__name__)

EGO_COLUMNS = (
    "txn_id, timestamp, src, dst, amt_usd, payment_format, pay_ccy, recv_ccy, is_laundering"
)


# --------------------------------------------------------------------------- construction
def build_multigraph(df: pd.DataFrame) -> nx.MultiDiGraph:
    """One edge per transaction (keyed by ``txn_id``).

    Edge attributes: ``timestamp, amt_usd, payment_format, is_laundering, txn_id``. The label is
    stored for evaluation and display only; it must never be used as a model feature.
    """
    g = nx.MultiDiGraph()
    cols = ["src", "dst", "txn_id", "timestamp", "amt_usd", "payment_format", "is_laundering"]
    g.add_edges_from(
        (s, d, t, {"txn_id": t, "timestamp": ts, "amt_usd": a, "payment_format": f,
                   "is_laundering": int(y)})
        for s, d, t, ts, a, f, y in df[cols].itertuples(index=False, name=None)
    )
    return g


def aggregate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transactions to one row per (src, dst) pair."""
    return (
        df.groupby(["src", "dst"], sort=False)
        .agg(total_usd=("amt_usd", "sum"), count=("txn_id", "size"),
             first_ts=("timestamp", "min"), last_ts=("timestamp", "max"),
             n_illicit=("is_laundering", "sum"))
        .reset_index()
    )


def build_aggregated_graph(df: pd.DataFrame, drop_self_loops: bool = False) -> nx.DiGraph:
    """``DiGraph`` with one edge per account pair: ``total_usd, count, first_ts, last_ts, n_illicit``."""
    if drop_self_loops:
        df = df[df["src"] != df["dst"]]
    pairs = aggregate_pairs(df)
    g = nx.DiGraph()
    g.add_edges_from(
        (s, d, {"total_usd": u, "count": int(c), "first_ts": f, "last_ts": l, "n_illicit": int(i)})
        for s, d, u, c, f, l, i in pairs.itertuples(index=False, name=None)
    )
    return g


# --------------------------------------------------------------------------- components
def strongly_connected_components(g: nx.DiGraph, min_size: int = 1) -> list[set[str]]:
    """SCCs of ``g`` with at least ``min_size`` nodes, largest first."""
    comps = [c for c in nx.strongly_connected_components(g) if len(c) >= min_size]
    return sorted(comps, key=len, reverse=True)


def weakly_connected_components(g: nx.DiGraph, min_size: int = 1) -> list[set[str]]:
    """WCCs of ``g`` with at least ``min_size`` nodes, largest first."""
    comps = [c for c in nx.weakly_connected_components(g) if len(c) >= min_size]
    return sorted(comps, key=len, reverse=True)


def nontrivial_scc_subgraph(g: nx.DiGraph) -> list[nx.DiGraph]:
    """Subgraphs induced by each SCC with more than one node (the only place cycles can live)."""
    return [g.subgraph(c).copy() for c in strongly_connected_components(g, min_size=2)]


def component_labels(
    src_idx: np.ndarray, dst_idx: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """SCC and WCC labels for a graph given as integer edge arrays (scipy, whole-graph scale)."""
    adj = sparse.coo_matrix(
        (np.ones(len(src_idx), dtype=np.int8), (src_idx, dst_idx)), shape=(n_nodes, n_nodes)
    ).tocsr()
    _, scc = csgraph.connected_components(adj, directed=True, connection="strong")
    _, wcc = csgraph.connected_components(adj, directed=True, connection="weak")
    return scc, wcc


# --------------------------------------------------------------------------- ego network
def ego_subgraph(
    account_id: str,
    k: int = 2,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    max_nodes: int | None = None,
    max_edges: int = 5000,
    parquet: Path | None = None,
    cfg: Config | None = None,
) -> nx.MultiDiGraph:
    """Time-filtered k-hop neighbourhood of ``account_id``, read from parquet via DuckDB.

    No in-memory full graph is needed. Neighbourhoods expand hop by hop (edges in either
    direction, self-loops ignored). If a hop would take the node count above ``max_nodes``, the
    new nodes are admitted in order of the largest USD value connecting them to the current
    frontier and the rest are dropped. The returned graph holds every transaction between the
    selected nodes in the window (at most ``max_edges``, highest value first).

    Node attributes: ``hop`` (distance from the centre), ``is_centre``. Edge attributes as in
    :func:`build_multigraph`. An unknown account yields an empty graph.
    """
    cfg = cfg or load_config()
    parquet = parquet or cfg.transactions_parquet
    max_nodes = max_nodes or cfg.graph.ego_max_nodes
    if k < 0:
        raise ValueError("k must be >= 0")
    path = parquet.as_posix()
    t0 = pd.Timestamp(start) if start is not None else None
    t1 = pd.Timestamp(end) if end is not None else None
    time_sql = ""
    params: list[object] = []
    if t0 is not None:
        time_sql += " AND timestamp >= ?"
        params.append(t0.to_pydatetime())
    if t1 is not None:
        time_sql += " AND timestamp <= ?"
        params.append(t1.to_pydatetime())

    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE TEMP VIEW tx AS SELECT {EGO_COLUMNS} FROM read_parquet('{path}') "
            f"WHERE src <> dst"
        )
        hop: dict[str, int] = {}
        exists = con.execute(
            "SELECT 1 FROM tx WHERE (src = ? OR dst = ?)" + time_sql + " LIMIT 1",
            [account_id, account_id, *params],
        ).fetchone()
        g = nx.MultiDiGraph()
        if not exists:
            logger.info("No transactions for %s in the selected window", account_id)
            return g
        hop[account_id] = 0
        frontier = [account_id]
        for depth in range(1, k + 1):
            con.register("frontier", pd.DataFrame({"acc": frontier}))
            cand = con.execute(
                f"""
                SELECT nb, max(amt_usd) AS best FROM (
                    SELECT dst AS nb, amt_usd FROM tx WHERE src IN (SELECT acc FROM frontier){time_sql}
                    UNION ALL
                    SELECT src AS nb, amt_usd FROM tx WHERE dst IN (SELECT acc FROM frontier){time_sql}
                ) GROUP BY nb
                """,
                params + params,
            ).df()
            con.unregister("frontier")
            cand = cand[~cand["nb"].isin(hop)].sort_values("best", ascending=False)
            room = max_nodes - len(hop)
            if room <= 0 or cand.empty:
                break
            if len(cand) > room:
                logger.info("Hop %d: pruning %d candidate nodes to %d", depth, len(cand), room)
                cand = cand.head(room)
            frontier = cand["nb"].tolist()
            hop.update({n: depth for n in frontier})

        con.register("nodes", pd.DataFrame({"acc": list(hop)}))
        edges = con.execute(
            f"""
            SELECT * FROM tx
            WHERE src IN (SELECT acc FROM nodes) AND dst IN (SELECT acc FROM nodes){time_sql}
            ORDER BY amt_usd DESC LIMIT {int(max_edges)}
            """,
            params,
        ).df()
    finally:
        con.close()

    g.add_nodes_from((n, {"hop": h, "is_centre": n == account_id}) for n, h in hop.items())
    g.add_edges_from(
        (r.src, r.dst, r.txn_id, {"txn_id": r.txn_id, "timestamp": r.timestamp,
                                   "amt_usd": r.amt_usd, "payment_format": r.payment_format,
                                   "pay_ccy": r.pay_ccy, "recv_ccy": r.recv_ccy,
                                   "is_laundering": int(r.is_laundering)})
        for r in edges.itertuples(index=False)
    )
    return g
