"""AML investigation dashboard.

Reads only precomputed files (``account_scores.parquet``, ``transaction_scores.parquet``,
``findings.parquet``) plus the saved model for on-demand SHAP explanations. Nothing is trained
here. Set ``AML_PROCESSED_DIR`` / ``AML_MODELS_DIR`` to point at other locations.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import colormaps
from matplotlib.colors import to_hex
from pyvis.network import Network

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml.graph import ego_subgraph  # noqa: E402

PROC = Path(os.environ.get("AML_PROCESSED_DIR", ROOT / "data" / "processed"))
MODELS = Path(os.environ.get("AML_MODELS_DIR", ROOT / "models"))
ANOMALY_CMAP, SCORE_CMAP = colormaps["YlOrRd"], colormaps["coolwarm"]
FINDING_STYLES = {  # detector family -> (label, vis.js dashes, colour)
    "temporal_cycle": ("Time-ordered cycle", [10, 6], "#e6550d"),
    "fan_out": ("Fan-out", [2, 6], "#756bb1"),
    "fan_in": ("Fan-in", [2, 6], "#756bb1"),
    "scatter_gather": ("Scatter-gather", [12, 4, 2, 4], "#31a354"),
    "gather_scatter": ("Gather-scatter", [12, 4, 2, 4], "#31a354"),
}

st.set_page_config(page_title="AML investigator", layout="wide")


# --------------------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Loading score tables...")
def load_tables() -> dict:
    accounts = pd.read_parquet(PROC / "account_scores.parquet")
    findings = pd.read_parquet(PROC / "findings.parquet")
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW txs AS SELECT * FROM read_parquet('{(PROC / 'transaction_scores.parquet').as_posix()}')")
    lo, hi = con.execute("SELECT min(timestamp), max(timestamp) FROM txs").fetchone()
    return {"accounts": accounts, "findings": findings, "con": con, "t_min": lo, "t_max": hi}


def account_txns(con, acc: str, start, end) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM txs WHERE (src = ? OR dst = ?) AND timestamp BETWEEN ? AND ? "
        "AND src <> dst ORDER BY timestamp",
        [acc, acc, start, end]).df()


def txn_scores(con, ids: list[int]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=["txn_id", "score"])
    con.register("ids", pd.DataFrame({"txn_id": ids}))
    return con.execute(
        "SELECT t.txn_id, t.score, t.flagged, t.split, t.is_laundering FROM txs t "
        "JOIN ids USING (txn_id)").df()


def short(acc: str) -> str:
    bank, _, a = acc.partition("_")
    return f"{bank}/{a[:6]}"


# --------------------------------------------------------------------------- network
def build_network_html(g: nx.MultiDiGraph, centre: str, tabs: dict, show_truth: bool) -> str:
    """Interactive pyvis network. Edges are aggregated per account pair."""
    acc = tabs["accounts"]
    con = tabs["con"]
    ids = [d["txn_id"] for _, _, d in g.edges(data=True)]
    sc = txn_scores(con, ids).set_index("txn_id")
    fin = tabs["findings"]
    in_graph = set(g.nodes)
    txn_det: dict[int, str] = {}
    for det, ids_, accs in zip(fin["detector"], fin["txn_ids"], fin["account_ids"]):
        if det in FINDING_STYLES and in_graph.intersection(accs[:50]):
            for t in ids_:
                txn_det.setdefault(int(t), det)

    flow = {n: 0.0 for n in g.nodes}
    pairs: dict[tuple[str, str], list[dict]] = {}
    for u, v, d in g.edges(data=True):
        flow[u] += d["amt_usd"]; flow[v] += d["amt_usd"]
        pairs.setdefault((u, v), []).append(d)
    max_flow = max(flow.values()) if flow else 1.0

    net = Network(height="620px", width="100%", directed=True, notebook=False,
                  cdn_resources="in_line", bgcolor="#ffffff")
    net.set_options(json.dumps({
        "physics": {"barnesHut": {"gravitationalConstant": -4500, "springLength": 120},
                    "stabilization": {"iterations": 120}},
        "interaction": {"hover": True, "tooltipDelay": 80},
        "edges": {"smooth": {"type": "continuous"}}}))
    illicit_nodes = {n for u, v, d in g.edges(data=True) if d.get("is_laundering")
                     for n in (u, v)} if show_truth else set()
    for n in g.nodes:
        row = acc.loc[n] if n in acc.index else None
        a_score = float(row["anomaly_score"]) if row is not None and pd.notna(row["anomaly_score"]) else 0.0
        size = 10 + 30 * np.log1p(flow[n]) / np.log1p(max_flow)
        is_c = n == centre
        title = (f"<b>{n}</b><br>anomaly score: {a_score:.3f}<br>"
                 f"flow in graph: ${flow[n]:,.0f}<br>hop: {g.nodes[n].get('hop', '?')}")
        if row is not None:
            title += f"<br>supervised max: {row['max_supervised_score']:.3f}"
        border = "#d62728" if n in illicit_nodes else ("#000000" if is_c else "#555555")
        net.add_node(n, label=short(n), title=title, size=size * (1.5 if is_c else 1.0),
                     shape="star" if is_c else "dot",
                     color={"background": to_hex(ANOMALY_CMAP(a_score)), "border": border},
                     borderWidth=4 if (is_c or n in illicit_nodes) else 1)
    for (u, v), items in pairs.items():
        total = sum(i["amt_usd"] for i in items)
        scores = [float(sc.loc[i["txn_id"], "score"]) if i["txn_id"] in sc.index else 0.0 for i in items]
        top = max(scores)
        dets = {txn_det[i["txn_id"]] for i in items if i["txn_id"] in txn_det}
        rows = "".join(
            f"<br>${i['amt_usd']:,.2f} | {pd.Timestamp(i['timestamp']):%Y-%m-%d %H:%M} | "
            f"{i['payment_format']} | {i['pay_ccy']}->{i['recv_ccy']}"
            for i in sorted(items, key=lambda i: -i["amt_usd"])[:6])
        title = f"<b>{len(items)} txn(s), ${total:,.0f}</b> (max score {top:.3f}){rows}"
        if dets:
            title += "<br><i>Findings: " + ", ".join(FINDING_STYLES[d][0] for d in sorted(dets)) + "</i>"
        if show_truth and any(i.get("is_laundering") for i in items):
            title += "<br><b>ground truth: laundering</b>"
        kw: dict = {}
        if dets:
            first = sorted(dets)[0]
            kw = {"dashes": FINDING_STYLES[first][1], "shadow": True}
        net.add_edge(u, v, width=1 + 5 * np.log1p(total) / np.log1p(max(1.0, max_flow)),
                     color=to_hex(SCORE_CMAP(min(1.0, top))), title=title, **kw)
    return net.generate_html(notebook=False)


# --------------------------------------------------------------------------- other panels
def sankey(con, acc: str, start, end, per_level: int = 8) -> go.Figure:
    """Money flowing in (up to two hops upstream) and out (up to two hops downstream)."""
    def top(direction: str, nodes: list[str]) -> pd.DataFrame:
        key, other = ("dst", "src") if direction == "in" else ("src", "dst")
        con.register("nodes_", pd.DataFrame({"n": nodes}))
        df = con.execute(
            f"SELECT {other} AS other, {key} AS node, sum(amt_usd) AS usd FROM txs "
            f"WHERE {key} IN (SELECT n FROM nodes_) AND src <> dst AND timestamp BETWEEN ? AND ? "
            f"GROUP BY 1, 2", [start, end]).df()
        return df.sort_values("usd", ascending=False).groupby("node").head(per_level)

    labels: dict[str, int] = {}
    src_i, dst_i, vals = [], [], []

    def nid(key: str) -> int:
        return labels.setdefault(key, len(labels))

    centre = nid(f"{acc}|C")
    up1 = top("in", [acc])
    for r in up1.itertuples():
        src_i.append(nid(f"{r.other}|in1")); dst_i.append(centre); vals.append(r.usd)
    up2 = top("in", list(up1["other"].unique())) if len(up1) else pd.DataFrame()
    for r in up2.itertuples():
        if r.other != acc:
            src_i.append(nid(f"{r.other}|in2")); dst_i.append(nid(f"{r.node}|in1")); vals.append(r.usd)
    dn1 = top("out", [acc])
    for r in dn1.itertuples():
        src_i.append(centre); dst_i.append(nid(f"{r.other}|out1")); vals.append(r.usd)
    dn2 = top("out", list(dn1["other"].unique())) if len(dn1) else pd.DataFrame()
    for r in dn2.itertuples():
        if r.other != acc:
            src_i.append(nid(f"{r.node}|out1")); dst_i.append(nid(f"{r.other}|out2")); vals.append(r.usd)
    names = [short(k.split("|")[0]) for k in labels]
    colours = ["#1f77b4" if k.endswith("|C") else "#9ecae1" if "|in" in k else "#fdae6b" for k in labels]
    fig = go.Figure(go.Sankey(
        node=dict(label=names, color=colours, pad=12, thickness=14),
        link=dict(source=src_i, target=dst_i, value=vals)))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
    return fig


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    import lightgbm as lgb

    meta = json.loads((MODELS / "lgbm_meta.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(MODELS / "lgbm.txt")), meta


@st.cache_data(show_spinner="Computing SHAP contributions...")
def shap_for_txns(txn_ids: tuple[int, ...]) -> pd.DataFrame:
    """SHAP values of the saved model for the given transactions (features read from parquet)."""
    import shap

    model, meta = load_model_bundle()
    con = duckdb.connect()
    ids = pd.DataFrame({"txn_id": list(txn_ids)})
    con.register("ids", ids)
    feats = con.execute(
        f"SELECT f.* FROM read_parquet('{(PROC / 'txn_features.parquet').as_posix()}') f "
        f"JOIN ids USING (txn_id)").df()
    x = feats[meta["features"]].copy()
    for c, cats in meta["categories"].items():
        x[c] = pd.Categorical(feats[c], categories=cats)
    sv = shap.TreeExplainer(model).shap_values(x)
    if isinstance(sv, list):
        sv = sv[1]
    out = pd.DataFrame(sv, columns=meta["features"])
    out.insert(0, "txn_id", feats["txn_id"].to_numpy())
    return out


# --------------------------------------------------------------------------- UI
def main() -> None:
    try:
        tabs = load_tables()
    except FileNotFoundError as exc:
        st.error(f"Score tables not found ({exc}). Run `python -m aml.cli train` first.")
        return
    acc_df: pd.DataFrame = tabs["accounts"]
    con = tabs["con"]
    st.title("AML investigation dashboard")
    st.caption("Synthetic IBM AML data. Scores are precomputed; this app never retrains.")

    # ---- sidebar
    def on_pick() -> None:
        sel = st.session_state.get("top_table", {}).get("selection", {}).get("rows", [])
        shown = st.session_state.get("top_df")
        if sel and shown is not None:
            st.session_state["account_input"] = shown.index[sel[0]]

    st.sidebar.header("Account")
    st.sidebar.text_input("Account ID (bank_account)", key="account_input",
                          placeholder="e.g. 070_100428660")
    st.sidebar.subheader("Top flagged accounts")
    n_top = st.sidebar.slider("Rows", 20, 500, 100, step=20)
    top = acc_df.sort_values("risk", ascending=False).head(n_top)[
        ["risk", "anomaly_score", "max_supervised_score"]].round(3)
    st.session_state["top_df"] = top
    st.sidebar.dataframe(top, key="top_table", on_select=on_pick, selection_mode="single-row",
                         height=300)
    st.sidebar.subheader("Neighbourhood")
    k = st.sidebar.slider("Hop depth k", 1, 3, 2)
    t_min, t_max = pd.Timestamp(tabs["t_min"]).to_pydatetime(), pd.Timestamp(tabs["t_max"]).to_pydatetime()
    start, end = st.sidebar.slider("Date range", min_value=t_min, max_value=t_max,
                                   value=(t_min, t_max), format="MMM DD HH:mm")
    max_nodes = st.sidebar.number_input("Max nodes", 20, 1000, 300, step=20)
    show_truth = st.sidebar.toggle("Show ground-truth labels", value=False,
                                   help="Off by default for blind investigation.")

    acc = (st.session_state.get("account_input") or "").strip()
    if not acc:
        st.info("Enter an account ID or pick one from the ranked table in the sidebar.")
        return
    if acc not in acc_df.index:
        st.warning(f"Unknown account `{acc}`. Account IDs look like `070_100428660` "
                   "(bank code, underscore, account).")
        return

    row = acc_df.loc[acc]
    fired = [c.removeprefix("hits_") for c in acc_df.columns if c.startswith("hits_") and row[c] > 0]

    # ---- header metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Anomaly score", f"{row['anomaly_score']:.3f}")
    c2.metric("Supervised score (max)", f"{row['max_supervised_score']:.3f}")
    c3.metric("Detectors fired", len(fired))
    c4.metric("Total inflow", f"${row['usd_in']:,.0f}")
    c5.metric("Total outflow", f"${row['usd_out']:,.0f}")
    st.write("**Detectors:** " + (", ".join(fired) if fired else "none"))
    if show_truth:
        st.write("**Ground truth:** " + ("account took part in laundering"
                                         if row["is_illicit_account"] else "no laundering transactions"))

    # ---- network
    st.subheader("Transaction network")
    g = ego_subgraph(acc, k=k, start=start, end=end, max_nodes=int(max_nodes),
                     parquet=PROC / "transactions.parquet")
    if g.number_of_edges() == 0:
        st.info("No transactions with other accounts in this date range.")
        return
    if g.number_of_nodes() >= max_nodes:
        st.caption(f"Neighbourhood capped at {max_nodes} nodes (highest-value links kept).")
    components.html(build_network_html(g, acc, tabs, show_truth), height=650, scrolling=False)
    st.caption("Node size = flow, colour = anomaly score (yellow low, red high), star = selected. "
               "Edge width = amount, colour = max supervised score (blue low, red high). "
               "Dashed styles: orange = cycle, purple dotted = fan-in/out, green dash-dot = "
               "scatter/gather.")

    # ---- sankey and timeline
    left, right = st.columns(2)
    with left:
        st.subheader("Money flow (two hops)")
        st.plotly_chart(sankey(con, acc, start, end), use_container_width=True)
    with right:
        st.subheader("Timeline")
        tx = account_txns(con, acc, start, end)
        tx["direction"] = np.where(tx["src"] == acc, "out", "in")
        tx["counterparty"] = np.where(tx["src"] == acc, tx["dst"], tx["src"])
        fig = px.scatter(tx, x="timestamp", y="amt_usd", color="score", symbol="direction",
                         color_continuous_scale="RdBu_r", range_color=(0, 1), log_y=True,
                         hover_data=["counterparty", "payment_format", "pay_ccy", "recv_ccy"])
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---- explanation
    st.subheader("Why was this account flagged?")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Top reasons**")
        for r in row["reasons"]:
            st.markdown(f"- {r}")
        fin = tabs["findings"]
        mine = fin[[acc in a for a in fin["account_ids"]]]
        if len(mine):
            st.markdown("**Rules that fired**")
            show = mine[["detector", "start_ts", "end_ts", "total_usd", "score", "detail"]]
            st.dataframe(show.sort_values("score", ascending=False).head(15), hide_index=True)
    with e2:
        st.markdown("**Top SHAP contributors** (highest-scored transactions)")
        if len(tx) and (MODELS / "lgbm.txt").exists():
            best = tuple(int(i) for i in tx.nlargest(20, "score")["txn_id"])
            try:
                sv = shap_for_txns(best)
                mean_abs = sv.drop(columns="txn_id").abs().mean().nlargest(10)[::-1]
                sig = sv.drop(columns="txn_id").mean()[mean_abs.index]
                bar = go.Figure(go.Bar(x=sig.values, y=sig.index, orientation="h",
                                       marker_color=np.where(sig.values > 0, "#d62728", "#1f77b4")))
                bar.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title="mean SHAP (log-odds)")
                st.plotly_chart(bar, use_container_width=True)
            except Exception as exc:  # explanation is optional; never break the page
                st.caption(f"SHAP unavailable: {exc}")
        else:
            st.caption("No transactions or no saved model.")
    splits = tx["split"].value_counts().to_dict() if len(tx) else {}
    st.caption(f"Transactions by split: {splits}. Scores on train/val rows are in-sample and "
               "optimistic; test rows are a fair evaluation.")


main()
