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
from aml.names import make_aliases  # noqa: E402

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

st.set_page_config(page_title="AML investigator", page_icon=":material/shield:", layout="wide")


# --------------------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Loading score tables...")
def load_tables() -> dict:
    accounts = pd.read_parquet(PROC / "account_scores.parquet")
    findings = pd.read_parquet(PROC / "findings.parquet")
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW txs AS SELECT * FROM read_parquet('{(PROC / 'transaction_scores.parquet').as_posix()}')")
    lo, hi = con.execute("SELECT min(timestamp), max(timestamp) FROM txs").fetchone()
    alias = make_aliases(accounts.index)
    by_name = {v.lower(): k for k, v in alias.items()}
    return {"accounts": accounts, "findings": findings, "con": con, "t_min": lo, "t_max": hi,
            "alias": alias, "by_name": by_name}


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
    """Readable name for an account (falls back to the raw ID for unknown accounts)."""
    return load_tables()["alias"].get(acc, acc)


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
        title = (f"<b>{short(n)}</b><br>{n}<br>anomaly score: {a_score:.3f}<br>"
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
def sankey(con, acc: str, start, end, per_level: int = 8, per_level2: int = 4) -> go.Figure:
    """Money flowing in (up to two hops upstream) and out (up to two hops downstream)."""
    def top(direction: str, nodes: list[str], limit: int) -> pd.DataFrame:
        key, other = ("dst", "src") if direction == "in" else ("src", "dst")
        con.register("nodes_", pd.DataFrame({"n": nodes}))
        df = con.execute(
            f"SELECT {other} AS other, {key} AS node, sum(amt_usd) AS usd FROM txs "
            f"WHERE {key} IN (SELECT n FROM nodes_) AND src <> dst AND timestamp BETWEEN ? AND ? "
            f"GROUP BY 1, 2", [start, end]).df()
        return df.sort_values("usd", ascending=False).groupby("node").head(limit)

    labels: dict[str, int] = {}
    src_i: list[int] = []
    dst_i: list[int] = []
    vals: list[float] = []
    link_col: list[str] = []
    IN_COL, OUT_COL = "rgba(46,144,250,0.28)", "rgba(247,144,9,0.30)"

    def nid(key: str) -> int:
        return labels.setdefault(key, len(labels))

    def link(a: str, b: str, v: float, colour: str) -> None:
        src_i.append(nid(a)); dst_i.append(nid(b)); vals.append(v); link_col.append(colour)

    centre = f"{acc}|C"
    nid(centre)
    up1 = top("in", [acc], per_level)
    for r in up1.itertuples():
        link(f"{r.other}|in1", centre, r.usd, IN_COL)
    up2 = top("in", list(up1["other"].unique()), per_level2) if len(up1) else pd.DataFrame()
    for r in up2.itertuples():
        if r.other != acc:
            link(f"{r.other}|in2", f"{r.node}|in1", r.usd, IN_COL)
    dn1 = top("out", [acc], per_level)
    for r in dn1.itertuples():
        link(centre, f"{r.other}|out1", r.usd, OUT_COL)
    dn2 = top("out", list(dn1["other"].unique()), per_level2) if len(dn1) else pd.DataFrame()
    for r in dn2.itertuples():
        if r.other != acc:
            link(f"{r.node}|out1", f"{r.other}|out2", r.usd, OUT_COL)
    names = [short(k.split("|")[0]) for k in labels]
    full = [k.split("|")[0] for k in labels]
    colours = ["#1b4965" if k.endswith("|C") else "#84caff" if "|in" in k else "#fec84b" for k in labels]
    fig = go.Figure(go.Sankey(
        node=dict(label=names, color=colours, pad=14, thickness=14, customdata=full,
                  hovertemplate="%{customdata}<br>$%{value:,.0f}<extra></extra>",
                  line=dict(color="#ffffff", width=0.5)),
        link=dict(source=src_i, target=dst_i, value=vals, color=link_col,
                  hovertemplate="$%{value:,.0f}<extra></extra>")))
    return style_fig(fig, 520)


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


# --------------------------------------------------------------------------- styling
CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1560px;}
header[data-testid="stHeader"] {background: transparent;}
h2, h3 {letter-spacing: -0.01em;}
.hero {background: linear-gradient(120deg, #0f2a43 0%, #1b4965 55%, #2a7f86 100%);
       color: #fff; border-radius: 18px; padding: 22px 30px; margin-bottom: 18px;
       box-shadow: 0 6px 18px rgba(15, 42, 67, .18);}
.hero h1 {color: #fff; margin: 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em;}
.hero p {color: #cfe3ee; margin: 6px 0 0; font-size: .95rem;}
.acct-strip {display: flex; align-items: center; flex-wrap: wrap; gap: 10px; margin: 2px 0 14px;}
.acct-id {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.25rem;
          font-weight: 650; color: #0f2a43;}
.muted {color: #667085; font-size: .85rem;}
.card {background: #fff; border: 1px solid #e3e8ee; border-radius: 14px; padding: 14px 16px 12px;
       box-shadow: 0 1px 2px rgba(16, 24, 40, .05); height: 100%;}
.card .lbl {font-size: .70rem; text-transform: uppercase; letter-spacing: .07em; color: #667085;
            font-weight: 600;}
.card .val {font-size: 1.75rem; font-weight: 680; color: #101828; line-height: 1.25;
            font-variant-numeric: tabular-nums;}
.card .sub {font-size: .78rem; color: #667085; margin-top: 1px;}
.bar {height: 6px; border-radius: 3px; background: #eef2f6; margin-top: 9px; overflow: hidden;}
.bar > span {display: block; height: 100%; border-radius: 3px;}
.chip {display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem;
       font-weight: 600; margin: 2px 6px 2px 0; border: 1px solid transparent;}
.badge {display: inline-block; padding: 3px 12px; border-radius: 8px; font-size: .78rem;
        font-weight: 700; letter-spacing: .03em;}
.panel-title {font-size: 1.08rem; font-weight: 650; color: #101828; margin: 26px 0 2px;}
.panel-sub {font-size: .85rem; color: #667085; margin-bottom: 8px;}
.legend {display: flex; flex-wrap: wrap; gap: 22px; align-items: center; font-size: .78rem;
         color: #475467; background: #fff; border: 1px solid #e3e8ee; border-radius: 12px;
         padding: 9px 16px; margin-top: 8px;}
.legend .grad {display: inline-block; width: 90px; height: 9px; border-radius: 5px;
               vertical-align: middle; margin: 0 6px;}
.legend svg {vertical-align: middle; margin-right: 5px;}
.welcome {background: #fff; border: 1px dashed #b8c4d1; border-radius: 16px; padding: 28px 32px;
          color: #344054;}
.welcome h3 {margin-top: 0;}
iframe {border-radius: 14px; border: 1px solid #e3e8ee;}
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e3e8ee;}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {font-size: .8rem; text-transform: uppercase;
        letter-spacing: .08em; color: #667085; margin-top: 1.2rem;}
</style>
"""
PLOT_FONT = dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12, color="#344054")
CHIP_COLOURS = {"temporal_cycle": "#e6550d", "fan_out": "#756bb1", "fan_in": "#756bb1",
                "scatter_gather": "#31a354", "gather_scatter": "#31a354", "pass_through": "#475467"}
DETECTOR_LABELS = {"temporal_cycle": "Cycle", "fan_out": "Fan-out", "fan_in": "Fan-in",
                   "scatter_gather": "Scatter-gather", "gather_scatter": "Gather-scatter",
                   "pass_through": "Pass-through"}


def fmt_usd(v: float) -> str:
    v = float(v)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:,.2f}{suffix}"
    return f"${v:,.0f}"


def card(label: str, value: str, sub: str = "", frac: float | None = None, colour: str = "#1d6f8a") -> str:
    bar = (f'<div class="bar"><span style="width:{max(0.0, min(1.0, frac)) * 100:.0f}%;'
           f'background:{colour}"></span></div>') if frac is not None else ""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div>{sub_html}{bar}</div>'


def risk_badge(risk_pct: float) -> str:
    if risk_pct >= 0.99:
        text, bg, fg = "HIGH RISK", "#fee4e2", "#b42318"
    elif risk_pct >= 0.90:
        text, bg, fg = "ELEVATED", "#fef0c7", "#b54708"
    else:
        text, bg, fg = "ROUTINE", "#d1fadf", "#05603a"
    return f'<span class="badge" style="background:{bg};color:{fg}">{text}</span>'


def legend_html() -> str:
    def line(dash: str, colour: str) -> str:
        return (f'<svg width="34" height="10"><line x1="0" y1="5" x2="34" y2="5" stroke="{colour}" '
                f'stroke-width="2.5" stroke-dasharray="{dash}"/></svg>')
    node_grad = "linear-gradient(90deg,#ffffcc,#fd8d3c,#800026)"
    edge_grad = "linear-gradient(90deg,#3b4cc0,#dddddd,#b40426)"
    return (
        '<div class="legend">'
        f'<span>Node colour (anomaly)<span class="grad" style="background:{node_grad}"></span>low &rarr; high</span>'
        f'<span>Edge colour (supervised)<span class="grad" style="background:{edge_grad}"></span>low &rarr; high</span>'
        '<span>Node size = flow &middot; Edge width = amount &middot; &#9733; = selected</span>'
        f'<span>{line("10 6", "#e6550d")}cycle</span>'
        f'<span>{line("2 6", "#756bb1")}fan-in / fan-out</span>'
        f'<span>{line("12 4 2 4", "#31a354")}scatter / gather</span>'
        '</div>')


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=8, b=4), font=PLOT_FONT,
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      hoverlabel=dict(bgcolor="#101828", font_color="#ffffff"))
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False, linecolor="#d0d5dd")
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f6", zeroline=False, linecolor="#d0d5dd")
    return fig


def panel(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="panel-title">{title}</div>' + (f'<div class="panel-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- UI
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        tabs = load_tables()
    except FileNotFoundError as exc:
        st.error(f"Score tables not found ({exc}). Run `python -m aml.cli train` first.")
        return
    acc_df: pd.DataFrame = tabs["accounts"]
    con = tabs["con"]
    st.markdown(
        '<div class="hero"><h1>AML investigation dashboard</h1>'
        '<p>Explore account neighbourhoods, detector findings and model scores on the synthetic IBM '
        'transactions dataset. Scores are precomputed; this app never retrains.</p></div>',
        unsafe_allow_html=True)

    # ---- sidebar
    def set_account(value: str) -> None:
        st.session_state["account_input"] = value

    def on_pick() -> None:
        sel = st.session_state.get("top_table", {}).get("selection", {}).get("rows", [])
        shown = st.session_state.get("top_df")
        if sel and shown is not None:
            st.session_state["account_input"] = shown.index[sel[0]]

    st.sidebar.markdown("## Account")
    st.sidebar.text_input("Account", key="account_input",
                          placeholder="name or ID, e.g. Amber Falcon 417 or 070_100428660",
                          label_visibility="collapsed")
    st.sidebar.markdown("### Top flagged accounts")
    n_top = st.sidebar.slider("Rows shown", 20, 500, 100, step=20)
    top = acc_df.sort_values("risk", ascending=False).head(n_top)[
        ["risk", "anomaly_score", "max_supervised_score"]]
    top = top.rename(columns={"risk": "Risk", "anomaly_score": "Anomaly", "max_supervised_score": "Supervised"})
    top.insert(0, "Name", [tabs["alias"][a] for a in top.index])
    st.session_state["top_df"] = top
    st.sidebar.dataframe(
        top, key="top_table", on_select=on_pick, selection_mode="single-row", height=330,
        column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f")
                       for c in top.columns if c != "Name"})
    st.sidebar.caption("Click a row to investigate it.")
    st.sidebar.markdown("### Neighbourhood")
    k = st.sidebar.slider("Hop depth", 1, 3, 2)
    t_min = pd.Timestamp(tabs["t_min"]).to_pydatetime()
    t_max = pd.Timestamp(tabs["t_max"]).to_pydatetime()
    start, end = st.sidebar.slider("Date range", min_value=t_min, max_value=t_max,
                                   value=(t_min, t_max), format="MMM DD HH:mm")
    max_nodes = st.sidebar.number_input("Maximum nodes", 20, 1000, 300, step=20)
    st.sidebar.markdown("### Display")
    show_truth = st.sidebar.toggle("Show ground-truth labels", value=False,
                                   help="Off by default so investigations stay blind.")

    acc = (st.session_state.get("account_input") or "").strip()
    acc = tabs["by_name"].get(acc.lower(), acc)  # accept a readable name or a raw ID
    if not acc:
        st.markdown('<div class="welcome"><h3>Start an investigation</h3>'
                    'Enter an account ID in the sidebar, click a row in the ranked table, or open one '
                    'of the highest-risk accounts below.</div>', unsafe_allow_html=True)
        st.write("")
        cols = st.columns(5)
        for col, name in zip(cols, acc_df.sort_values("risk", ascending=False).index[:5]):
            col.button(tabs["alias"][name], on_click=set_account, args=(name,), width="stretch")
        return
    if acc not in acc_df.index:
        st.warning(f"Unknown account `{acc}`. Use a name like `Amber Falcon 417` "
                   "or an ID like `070_100428660` (bank code, underscore, account number).")
        return

    row = acc_df.loc[acc]
    fired = [c.removeprefix("hits_") for c in acc_df.columns if c.startswith("hits_") and row[c] > 0]
    risk_pct = float((acc_df["risk"] < row["risk"]).mean())
    anom_pct = float((acc_df["anomaly_score"] < row["anomaly_score"]).mean())
    bank = acc.partition("_")[0]

    # ---- header
    st.markdown(
        f'<div class="acct-strip"><span class="acct-id">{short(acc)}</span>{risk_badge(risk_pct)}'
        f'<span class="muted">ID {acc} &middot; bank {bank} &middot; {int(row["n_in"])} incoming / {int(row["n_out"])} '
        f'outgoing transactions &middot; risk percentile {risk_pct * 100:.1f}</span></div>',
        unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(card("Anomaly score", f"{row['anomaly_score']:.3f}", f"higher than {anom_pct * 100:.1f}% of accounts",
                     row["anomaly_score"], to_hex(ANOMALY_CMAP(row["anomaly_score"]))), unsafe_allow_html=True)
    c2.markdown(card("Supervised score (max)", f"{row['max_supervised_score']:.3f}",
                     "percentile of strongest transaction", row["max_supervised_score"],
                     to_hex(SCORE_CMAP(row["max_supervised_score"]))), unsafe_allow_html=True)
    c3.markdown(card("Detectors fired", str(len(fired)), "of 6 rule-based detectors",
                     len(fired) / 6, "#6941c6"), unsafe_allow_html=True)
    c4.markdown(card("Total inflow", fmt_usd(row["usd_in"]), "USD received"), unsafe_allow_html=True)
    c5.markdown(card("Total outflow", fmt_usd(row["usd_out"]), "USD sent"), unsafe_allow_html=True)
    chips = "".join(
        f'<span class="chip" style="background:{CHIP_COLOURS[d]}1a;color:{CHIP_COLOURS[d]};'
        f'border-color:{CHIP_COLOURS[d]}55">{DETECTOR_LABELS[d]} &times;{int(row["hits_" + d])}</span>'
        for d in fired)
    st.markdown(f'<div style="margin-top:12px">{chips or "<span class=muted>No rule-based detector fired.</span>"}</div>',
                unsafe_allow_html=True)
    if show_truth:
        verdict = "took part in laundering" if row["is_illicit_account"] else "no laundering transactions"
        st.info(f"Ground truth: this account {verdict}.")

    # ---- network
    panel("Transaction network", f"{k}-hop neighbourhood, {start:%b %d} to {end:%b %d}")
    g = ego_subgraph(acc, k=k, start=start, end=end, max_nodes=int(max_nodes),
                     parquet=PROC / "transactions.parquet")
    if g.number_of_edges() == 0:
        st.info("No transactions with other accounts in this date range. Widen the range in the sidebar.")
        return
    if g.number_of_nodes() >= max_nodes:
        st.caption(f"Capped at {max_nodes} nodes; the highest-value links were kept.")
    components.html(build_network_html(g, acc, tabs, show_truth), height=650, scrolling=False)
    st.markdown(legend_html(), unsafe_allow_html=True)

    # ---- sankey and timeline
    left, right = st.columns(2, gap="large")
    with left:
        panel("Money flow", "Largest flows into and out of the account, up to two hops each way")
        st.plotly_chart(sankey(con, acc, start, end), width="stretch")
    tx = account_txns(con, acc, start, end)
    with right:
        panel("Timeline", "Each point is a transaction; colour is the supervised score")
        tx["direction"] = np.where(tx["src"] == acc, "out", "in")
        tx["counterparty"] = np.where(tx["src"] == acc, tx["dst"], tx["src"])
        tx["counterparty"] = tx["counterparty"].map(short)
        fig = px.scatter(tx, x="timestamp", y="amt_usd", color="score", symbol="direction",
                         color_continuous_scale="RdBu_r", range_color=(0, 1), log_y=True,
                         hover_data=["counterparty", "payment_format", "pay_ccy", "recv_ccy"],
                         labels={"amt_usd": "Amount (USD)", "timestamp": "", "score": "Score"})
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#ffffff")))
        fig.update_layout(coloraxis_colorbar=dict(thickness=10, len=0.8, title=""),
                          legend=dict(orientation="h", y=1.08, x=0, title=""))
        st.plotly_chart(style_fig(fig, 520), width="stretch")

    # ---- explanation
    panel("Why was this account flagged?", "Rules, standout features and model drivers")
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.markdown("**Top reasons**")
        for r in row["reasons"]:
            st.markdown(f"- {r}")
        fin = tabs["findings"]
        mine = fin[[acc in a for a in fin["account_ids"]]]
        if len(mine):
            st.markdown("**Rules that fired**")
            show = mine[["detector", "start_ts", "end_ts", "total_usd", "score", "detail"]].copy()
            show["detector"] = show["detector"].map(DETECTOR_LABELS)
            st.dataframe(show.sort_values("score", ascending=False).head(15), hide_index=True,
                         column_config={"score": st.column_config.ProgressColumn(
                             "Score", min_value=0.0, max_value=1.0, format="%.2f"),
                             "total_usd": st.column_config.NumberColumn("USD", format="$%,.0f")})
        else:
            st.caption("No detector findings involve this account.")
    with e2:
        st.markdown("**Top SHAP contributors** (account's 20 highest-scored transactions)")
        if len(tx) and (MODELS / "lgbm.txt").exists():
            best = tuple(int(i) for i in tx.nlargest(20, "score")["txn_id"])
            try:
                sv = shap_for_txns(best).drop(columns="txn_id")
                order = sv.abs().mean().nlargest(10)[::-1].index
                sig = sv.mean()[order]
                bar = go.Figure(go.Bar(
                    x=sig.values, y=[c.replace("_", " ") for c in sig.index], orientation="h",
                    marker_color=np.where(sig.values > 0, "#d92d20", "#2e90fa")))
                bar.update_layout(xaxis_title="mean SHAP value (log-odds); red raises risk")
                st.plotly_chart(style_fig(bar, 380), width="stretch")
            except Exception as exc:  # explanation is optional; never break the page
                st.caption(f"SHAP unavailable: {exc}")
        else:
            st.caption("No transactions in range, or no saved model.")
    splits = tx["split"].value_counts().to_dict() if len(tx) else {}
    st.caption(f"Transactions by split: {splits}. Scores on train and validation rows are in-sample "
               "and optimistic; only test rows are a fair evaluation.")


main()
