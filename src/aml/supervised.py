"""Supervised transaction-level baseline: LightGBM with a strictly temporal split (Phase 5).

Leakage controls:
* features come only from :func:`aml.features.build_transaction_features` (history strictly
  before each transaction); the label column is never in the feature list;
* the split is by time: first 60% of transactions train, next 20% validation, last 20% test;
* early stopping and the decision threshold use the validation set only;
* Phase 3 findings and Phase 4 account features are *not* used: they were computed over the whole
  period, i.e. with information from after the split boundaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve

from aml.config import Config
from aml.evaluate import precision_at_k
from aml.features import CATEGORICAL_FEATURES, TXN_FEATURES_FILE

logger = logging.getLogger(__name__)

NON_FEATURES = ["txn_id", "timestamp", "src", "dst", "is_laundering"]
MODEL_FILE = "lgbm.txt"
META_FILE = "lgbm_meta.json"
ERA_CUT = pd.Timestamp("2022-09-11")  # licit activity collapses after ~10 Sept (see EDA)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers, timestamp and the label."""
    return [c for c in df.columns if c not in NON_FEATURES]


def temporal_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[np.ndarray, dict[str, pd.Timestamp]]:
    """Assign each row to 0=train, 1=validation, 2=test by time order of its position.

    ``df`` must already be sorted by (timestamp, txn_id). Returns the split codes and the first
    timestamp of validation and test.
    """
    n = len(df)
    i_val, i_test = int(n * train_frac), int(n * (train_frac + val_frac))
    codes = np.zeros(n, dtype="int8")
    codes[i_val:] = 1
    codes[i_test:] = 2
    ts = df["timestamp"]
    bounds = {"train_start": ts.iloc[0], "val_start": ts.iloc[i_val],
              "test_start": ts.iloc[i_test], "test_end": ts.iloc[-1]}
    return codes, bounds


def prepare_features(df: pd.DataFrame, categories: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Feature matrix with categoricals as pandas ``category`` (fixed levels if provided)."""
    x = df[feature_columns(df)].copy()
    for col in CATEGORICAL_FEATURES:
        cats = categories[col] if categories else sorted(df[col].dropna().unique())
        x[col] = pd.Categorical(df[col], categories=cats)
    return x


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising the minority-class F1 (and that F1)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float, ks: tuple[int, ...]) -> dict[str, float]:
    """F1, precision, recall, PR-AUC, confusion matrix cells and precision@k at ``thr``."""
    pred = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "n": len(y), "n_positive": int(y.sum()), "threshold": thr,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(y, p, k)
    return out


def train_supervised(cfg: Config) -> dict:
    """Train, tune the threshold on validation, evaluate on test, and save model and reports."""
    proc = cfg.root / cfg.paths.processed_dir
    df = pd.read_parquet(proc / TXN_FEATURES_FILE)
    df = df.sort_values(["timestamp", "txn_id"], kind="stable").reset_index(drop=True)
    codes, bounds = temporal_split(df, cfg.supervised.train_frac, cfg.supervised.val_frac)
    logger.info("Split boundaries: %s", {k: str(v) for k, v in bounds.items()})
    cats = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL_FEATURES}
    x = prepare_features(df, cats)
    y = df["is_laundering"].to_numpy()
    tr, va, te = codes == 0, codes == 1, codes == 2
    for name, m in (("train", tr), ("val", va), ("test", te)):
        logger.info("%s: %d rows, %d laundering (%.4f%%)", name, m.sum(), y[m].sum(),
                    100 * y[m].mean())

    spw = float((y[tr] == 0).sum() / max(y[tr].sum(), 1))
    params = {
        "objective": "binary", "metric": "average_precision", "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 50, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "scale_pos_weight": spw,
        "seed": cfg.seed, "verbosity": -1, "num_threads": -1,
    }
    dtrain = lgb.Dataset(x[tr], y[tr])
    dval = lgb.Dataset(x[va], y[va], reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=1000, valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    logger.info("Best iteration %d (val AP %.4f)", model.best_iteration,
                model.best_score["val"]["average_precision"])

    p_val = model.predict(x[va], num_iteration=model.best_iteration)
    p_te = model.predict(x[te], num_iteration=model.best_iteration)
    thr, f1_val = best_f1_threshold(y[va], p_val)
    logger.info("Threshold %.4f (validation F1 %.4f)", thr, f1_val)
    ks = tuple(cfg.anomaly.precision_at_k)
    res = {"validation": metrics_at(y[va], p_val, thr, ks), "test": metrics_at(y[te], p_te, thr, ks)}

    # Era check: the test window straddles the late period where licit activity collapses.
    ts_te = df.loc[te, "timestamp"].to_numpy()
    for label, mask in (("test_before_11sep", ts_te < np.datetime64(ERA_CUT)),
                        ("test_from_11sep", ts_te >= np.datetime64(ERA_CUT))):
        if mask.any() and y[te][mask].sum() > 0 and (y[te][mask] == 0).any():
            res[label] = metrics_at(y[te][mask], p_te[mask], thr, ks)

    tables = cfg.root / cfg.paths.tables_dir
    pd.DataFrame(res).T.to_csv(tables / "supervised_metrics.csv")
    imp = pd.Series(model.feature_importance("gain"), index=x.columns, name="gain")
    imp.sort_values(ascending=False).to_csv(tables / "supervised_feature_importance.csv")

    typ = recall_by_typology(df.loc[te, ["txn_id"]].assign(p=p_te, y=y[te]), proc, thr)
    typ.to_csv(tables / "supervised_recall_by_typology.csv")
    logger.info("Test recall by typology:\n%s", typ.round(3).to_string())

    pd.DataFrame({"txn_id": df["txn_id"], "split": np.array(["train", "val", "test"])[codes]}
                 ).to_parquet(proc / "txn_split.parquet", index=False)
    models = cfg.root / cfg.paths.models_dir
    models.mkdir(parents=True, exist_ok=True)
    model.save_model(str(models / MODEL_FILE), num_iteration=model.best_iteration)
    meta = {"features": list(x.columns), "categories": cats, "threshold": thr,
            "best_iteration": model.best_iteration, "scale_pos_weight": spw,
            "boundaries": {k: str(v) for k, v in bounds.items()}}
    (models / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    shap_summary_plot(model, x[te], y[te], cfg.root / cfg.paths.figures_dir / "shap_summary.png",
                      cfg.seed)
    logger.info("Test metrics: %s", {k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in res["test"].items()})
    return res


def recall_by_typology(test: pd.DataFrame, proc: Path, thr: float) -> pd.DataFrame:
    """Share of test-set laundering transactions flagged (``p >= thr``), by attempt typology.

    ``UNATTRIBUTED`` collects laundering transactions that are in no listed attempt.
    """
    pat = pd.read_parquet(proc / "patterns.parquet", columns=["txn_id", "typology"]).dropna()
    pat["txn_id"] = pat["txn_id"].astype("int64")
    pos = test[test["y"] == 1].merge(pat, on="txn_id", how="left")
    pos["typology"] = pos["typology"].fillna("UNATTRIBUTED")
    pos["flagged"] = pos["p"] >= thr
    out = pos.groupby("typology").agg(n_test_txns=("flagged", "size"), recall=("flagged", "mean"))
    out.loc["ALL"] = [len(pos), pos["flagged"].mean()]
    out["n_test_txns"] = out["n_test_txns"].astype(int)
    return out


def shap_summary_plot(model: lgb.Booster, x: pd.DataFrame, y: np.ndarray, path: Path,
                      seed: int, n_neg: int = 4000, n_pos: int = 2000) -> None:
    """SHAP beeswarm on a sample (all/most positives plus random negatives) of ``x``."""
    import matplotlib.pyplot as plt
    import shap

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    idx = np.concatenate([rng.choice(pos, min(n_pos, len(pos)), replace=False),
                          rng.choice(neg, min(n_neg, len(neg)), replace=False)])
    sample = x.iloc[idx]
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):  # older shap returns [neg, pos]
        sv = sv[1]
    shap.summary_plot(sv, sample, max_display=20, show=False)
    plt.title("SHAP summary (test sample: positives + random negatives)")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def load_model(cfg: Config) -> tuple[lgb.Booster, dict]:
    """Load the saved booster and its metadata."""
    models = cfg.root / cfg.paths.models_dir
    meta = json.loads((models / META_FILE).read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(models / MODEL_FILE)), meta
