"""Tests for graph construction, components and ego-network extraction on a hand-made graph."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aml.graph import (
    build_aggregated_graph, build_multigraph, component_labels, ego_subgraph,
    nontrivial_scc_subgraph, strongly_connected_components, weakly_connected_components,
)

# A->B->C->A is a cycle; D->A feeds it; E->F is separate; A->A is a self-loop;
# G->A happens late (2022-09-10) and is the only edge outside the early window.
ROWS = [
    # src, dst, ts, usd, label
    ("A", "B", "2022-09-01 10:00", 100.0, 1),
    ("B", "C", "2022-09-01 11:00", 95.0, 1),
    ("C", "A", "2022-09-01 12:00", 90.0, 1),
    ("C", "A", "2022-09-02 12:00", 10.0, 0),
    ("D", "A", "2022-09-01 09:00", 500.0, 0),
    ("E", "F", "2022-09-03 09:00", 50.0, 0),
    ("A", "A", "2022-09-04 09:00", 5.0, 0),
    ("G", "A", "2022-09-10 09:00", 1.0, 0),
]


@pytest.fixture()
def txns() -> pd.DataFrame:
    df = pd.DataFrame(ROWS, columns=["src", "dst", "timestamp", "amt_usd", "is_laundering"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["txn_id"] = np.arange(len(df), dtype="int64")
    df["payment_format"] = "ACH"
    df["pay_ccy"] = df["recv_ccy"] = "US Dollar"
    return df


@pytest.fixture()
def parquet(txns: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "t.parquet"
    txns.to_parquet(path, index=False)
    return path


def test_multigraph_keeps_every_transaction(txns: pd.DataFrame) -> None:
    g = build_multigraph(txns)
    assert g.number_of_edges() == len(txns)
    assert g.number_of_edges("C", "A") == 2
    assert g["A"]["B"][0]["amt_usd"] == 100.0


def test_aggregated_graph_attributes(txns: pd.DataFrame) -> None:
    g = build_aggregated_graph(txns)
    assert g.number_of_edges() == 7  # C->A collapsed
    e = g["C"]["A"]
    assert e["count"] == 2 and e["total_usd"] == pytest.approx(100.0) and e["n_illicit"] == 1
    assert e["first_ts"] == pd.Timestamp("2022-09-01 12:00")
    assert e["last_ts"] == pd.Timestamp("2022-09-02 12:00")
    assert not build_aggregated_graph(txns, drop_self_loops=True).has_edge("A", "A")


def test_components(txns: pd.DataFrame) -> None:
    g = build_aggregated_graph(txns, drop_self_loops=True)
    assert strongly_connected_components(g, min_size=2) == [{"A", "B", "C"}]
    wcc = weakly_connected_components(g)
    assert wcc[0] == {"A", "B", "C", "D", "G"} and wcc[1] == {"E", "F"}
    subs = nontrivial_scc_subgraph(g)
    assert len(subs) == 1 and set(subs[0].nodes) == {"A", "B", "C"}


def test_component_labels_scipy_matches_networkx() -> None:
    # nodes 0..4: 0->1->2->0 cycle, 3->0, 4 isolated
    src, dst = np.array([0, 1, 2, 3]), np.array([1, 2, 0, 0])
    scc, wcc = component_labels(src, dst, 5)
    assert scc[0] == scc[1] == scc[2] and scc[3] != scc[0] and scc[4] != scc[0]
    assert wcc[0] == wcc[3] and wcc[4] != wcc[0]


def test_ego_hops(parquet: Path) -> None:
    g1 = ego_subgraph("D", k=1, parquet=parquet, max_nodes=50)
    assert set(g1.nodes) == {"D", "A"}
    g2 = ego_subgraph("D", k=2, parquet=parquet, max_nodes=50)
    assert set(g2.nodes) == {"D", "A", "B", "C", "G"}
    assert g2.nodes["D"]["is_centre"] and g2.nodes["B"]["hop"] == 2
    assert ego_subgraph("E", k=3, parquet=parquet, max_nodes=50).number_of_nodes() == 2
    assert not any(u == v for u, v in g2.edges())  # self-loops excluded


def test_ego_time_window(parquet: Path) -> None:
    g = ego_subgraph("A", k=1, start="2022-09-01", end="2022-09-02 23:59", parquet=parquet,
                     max_nodes=50)
    assert "G" not in g.nodes  # its only edge is on 2022-09-10
    assert {"B", "C", "D"} <= set(g.nodes)


def test_ego_prunes_to_highest_value(parquet: Path) -> None:
    g = ego_subgraph("A", k=1, parquet=parquet, max_nodes=3)
    assert g.number_of_nodes() == 3
    assert "D" in g.nodes  # largest-value neighbour (500) always survives pruning
    assert "G" not in g.nodes  # smallest-value neighbour (1) is dropped


def test_ego_unknown_account_is_empty(parquet: Path) -> None:
    g = ego_subgraph("NOPE", k=2, parquet=parquet, max_nodes=50)
    assert g.number_of_nodes() == 0
