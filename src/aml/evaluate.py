"""Evaluation helpers: typology recall for rule-based detectors (Phase 3).

Labels are used here, and only here, to *evaluate* detectors; the detectors never see them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aml.config import Config

logger = logging.getLogger(__name__)


def txn_sets(findings: pd.DataFrame) -> dict[str, set[int]]:
    """Union of transaction ids covered by each detector."""
    out: dict[str, set[int]] = {}
    for name, grp in findings.groupby("detector"):
        out[name] = {int(t) for ids in grp["txn_ids"] for t in ids}
    return out


def typology_recall(
    findings: pd.DataFrame,
    patterns: pd.DataFrame,
    threshold: float = 0.5,
    detectors: list[str] | None = None,
) -> pd.DataFrame:
    """Fraction of laundering attempts detected, per typology (rows) and detector (columns).

    An attempt counts as detected by a detector when at least ``threshold`` of its transactions
    appear in that detector's findings. ``any_detector`` uses the union over all detectors.
    ``n_attempts`` is the number of attempts of that typology; the ``ALL`` row pools them.
    """
    sets = txn_sets(findings)
    detectors = detectors or sorted(sets)
    pat = patterns.dropna(subset=["txn_id"]).copy()
    pat["txn_id"] = pat["txn_id"].astype("int64")
    union = set().union(*sets.values()) if sets else set()
    rows = []
    for attempt_id, grp in pat.groupby("attempt_id"):
        ids = set(grp["txn_id"])
        rec = {"attempt_id": attempt_id, "typology": grp["typology"].iloc[0]}
        for d in detectors:
            rec[d] = len(ids & sets.get(d, set())) / len(ids) >= threshold
        rec["any_detector"] = len(ids & union) / len(ids) >= threshold
        rows.append(rec)
    per_attempt = pd.DataFrame(rows)
    cols = [*detectors, "any_detector"]
    table = per_attempt.groupby("typology")[cols].mean()
    table.insert(0, "n_attempts", per_attempt.groupby("typology").size())
    table.loc["ALL"] = [len(per_attempt), *per_attempt[cols].mean()]
    table["n_attempts"] = table["n_attempts"].astype(int)
    return table


def detector_precision(
    findings: pd.DataFrame, labels: pd.Series, patterns: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Transaction-level precision and recall per detector.

    ``labels`` is ``is_laundering`` indexed by ``txn_id``. Columns: ``n_findings``,
    ``n_txns`` (distinct flagged), ``n_laundering``, ``precision`` (share of flagged transactions
    that are laundering), ``recall_all`` (share of all laundering transactions flagged),
    ``recall_attributed`` (same, restricted to transactions that belong to a listed attempt) and
    ``finding_precision`` (share of findings in which >= ``threshold`` of transactions are
    laundering).
    """
    sets = txn_sets(findings)
    lab = labels.astype(bool)
    laundering = set(lab.index[lab])
    attributed = set(patterns["txn_id"].dropna().astype("int64")) & laundering
    rows = []
    for name, grp in findings.groupby("detector"):
        flagged = sets[name]
        hit = flagged & laundering
        frac = grp["txn_ids"].map(lambda ids: float(np.mean([t in laundering for t in ids])))
        rows.append({
            "detector": name, "n_findings": len(grp), "n_txns": len(flagged),
            "n_laundering": len(hit),
            "precision": len(hit) / len(flagged) if flagged else np.nan,
            "recall_all": len(hit) / len(laundering),
            "recall_attributed": len(flagged & attributed) / len(attributed),
            "finding_precision": float((frac >= threshold).mean()),
        })
    union = set().union(*sets.values()) if sets else set()
    rows.append({
        "detector": "ANY", "n_findings": len(findings), "n_txns": len(union),
        "n_laundering": len(union & laundering),
        "precision": len(union & laundering) / len(union) if union else np.nan,
        "recall_all": len(union & laundering) / len(laundering),
        "recall_attributed": len(union & attributed) / len(attributed),
        "finding_precision": np.nan,
    })
    return pd.DataFrame(rows).set_index("detector")


def recall_by_attempt_size(
    findings: pd.DataFrame, patterns: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Share of attempts detected by *any* detector, by number of transactions in the attempt.

    Small attempts (1-2 transactions) are easily covered by coincidence, so this separates
    genuine structural recall from incidental coverage.
    """
    union = set().union(*txn_sets(findings).values()) if len(findings) else set()
    pat = patterns.dropna(subset=["txn_id"])
    rows = [
        {"attempt_id": a, "size": len(g),
         "detected": len(set(g["txn_id"].astype("int64")) & union) / len(g) >= threshold}
        for a, g in pat.groupby("attempt_id")
    ]
    df = pd.DataFrame(rows)
    df["size_bucket"] = pd.cut(df["size"], [0, 2, 5, 10, 1000], labels=["1-2", "3-5", "6-10", ">10"])
    return df.groupby("size_bucket", observed=True).agg(
        attempts=("attempt_id", "size"), detected=("detected", "mean"))


def cycle_recall_by_hops(
    findings: pd.DataFrame, patterns: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Recall of the temporal-cycle detector on CYCLE attempts, bucketed by the attempt's hop count."""
    found = txn_sets(findings).get("temporal_cycle", set())
    cyc = patterns[patterns["typology"] == "CYCLE"].dropna(subset=["txn_id"])
    rows = []
    for a, g in cyc.groupby("attempt_id"):
        hops = int(g["detail"].str.extract(r"Max (\d+)")[0].iloc[0])
        ids = set(g["txn_id"].astype("int64"))
        rows.append({"attempt_id": a, "hops": hops, "detected": len(ids & found) / len(ids) >= threshold})
    df = pd.DataFrame(rows)
    df["hop_bucket"] = pd.cut(df["hops"], [0, 2, 3, 6, 100], labels=["<=2", "3", "4-6", ">6"])
    return df.groupby("hop_bucket", observed=True).agg(
        attempts=("attempt_id", "size"), detected=("detected", "mean"))


def evaluate_detectors(cfg: Config, findings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute and save the typology x detector recall and the precision tables."""
    patterns = pd.read_parquet(cfg.root / cfg.paths.processed_dir / "patterns.parquet")
    labels = pd.read_parquet(cfg.transactions_parquet, columns=["txn_id", "is_laundering"])
    labels = labels.set_index("txn_id")["is_laundering"]
    thr = cfg.detectors.attempt_detection_threshold
    recall = typology_recall(findings, patterns, thr)
    prec = detector_precision(findings, labels, patterns, thr)
    tables = cfg.root / cfg.paths.tables_dir
    recall.to_csv(tables / "detector_recall_by_typology.csv")
    recall_by_attempt_size(findings, patterns, thr).to_csv(tables / "detector_recall_by_attempt_size.csv")
    cycle_recall_by_hops(findings, patterns, thr).to_csv(tables / "detector_cycle_recall_by_hops.csv")
    prec.to_csv(tables / "detector_precision.csv")
    logger.info("Typology recall:\n%s", recall.round(3).to_string())
    logger.info("Detector precision:\n%s", prec.round(4).to_string())
    return recall, prec
