"""Unsupervised account anomaly detection: Isolation Forest, HDBSCAN, LOF and a rank ensemble.

All scores are "higher = more anomalous". No label is used anywhere in this module.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from aml.config import AnomalyConfig, Config
from aml.features import add_detector_counts, transform_features

logger = logging.getLogger(__name__)

SCORE_COLUMNS = ["iforest", "lof", "hdbscan"]
ANOMALY_SCORES_FILE = "anomaly_scores.parquet"


def isolation_forest_scores(x: np.ndarray, cfg: AnomalyConfig, seed: int) -> np.ndarray:
    """Isolation Forest anomaly score on the full standardised feature matrix."""
    model = IsolationForest(n_estimators=cfg.iforest_estimators, random_state=seed, n_jobs=-1)
    model.fit(x)
    return -model.score_samples(x)


def lof_scores(z: np.ndarray, cfg: AnomalyConfig) -> np.ndarray:
    """Local Outlier Factor (in the PCA subspace); larger = more outlying."""
    lof = LocalOutlierFactor(n_neighbors=cfg.lof_neighbors, n_jobs=-1)
    lof.fit(z)
    return -lof.negative_outlier_factor_


def hdbscan_scores(z: np.ndarray, cfg: AnomalyConfig) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN-based anomaly score and binary flag.

    Noise points and members of clusters smaller than ``small_cluster_size`` are flagged
    anomalous. The score is ``1 - membership probability`` (so noise scores 1), raised to at
    least 0.75 for members of small clusters. Many points tie at 1.0 (all noise), so metrics for
    this score are tie-aware (see :func:`aml.evaluate.precision_at_k`).
    """
    model = HDBSCAN(min_cluster_size=cfg.min_cluster_size, n_jobs=-1, copy=True)
    model.fit(z)
    labels = model.labels_
    sizes = pd.Series(labels[labels >= 0]).value_counts()
    small = set(sizes[sizes < cfg.small_cluster_size].index)
    in_small = np.isin(labels, list(small))
    noise = labels == -1
    score = 1.0 - model.probabilities_
    score[noise] = 1.0
    score[in_small] = np.maximum(score[in_small], 0.75)
    flag = noise | in_small
    logger.info(
        "HDBSCAN: %d clusters (%d small), %.1f%% noise, %.1f%% flagged",
        len(sizes), len(small), 100 * noise.mean(), 100 * flag.mean(),
    )
    return score, flag


def mean_rank(scores: pd.DataFrame) -> pd.Series:
    """Mean of per-column normalised ranks (ties averaged); in (0, 1], higher = more anomalous."""
    ranks = scores.apply(lambda c: rankdata(c, method="average") / len(c))
    return ranks.mean(axis=1)


def fit_anomaly_scores(feats: pd.DataFrame, cfg: AnomalyConfig, seed: int) -> pd.DataFrame:
    """Run the three detectors and the rank ensemble; returns a frame indexed like ``feats``.

    Columns: ``iforest, lof, hdbscan, hdbscan_flag, ensemble``.
    """
    x, _ = transform_features(feats)
    z = PCA(n_components=cfg.pca_components, random_state=seed).fit_transform(x)
    logger.info("Feature matrix %s; PCA subspace %s", x.shape, z.shape)
    out = pd.DataFrame(index=feats.index)
    out["iforest"] = isolation_forest_scores(x, cfg, seed)
    logger.info("Isolation Forest done")
    out["lof"] = lof_scores(z, cfg)
    logger.info("LOF done")
    out["hdbscan"], out["hdbscan_flag"] = hdbscan_scores(z, cfg)
    out["ensemble"] = mean_rank(out[SCORE_COLUMNS])
    return out


def run_anomaly(cfg: Config, findings: pd.DataFrame) -> pd.DataFrame:
    """Load account features, add detector counts, fit all models and save the scores."""
    proc = cfg.root / cfg.paths.processed_dir
    feats = pd.read_parquet(proc / "account_features.parquet")
    feats = add_detector_counts(feats, findings)
    feats.to_parquet(proc / "account_features_full.parquet")
    scores = fit_anomaly_scores(feats, cfg.anomaly, cfg.seed)
    scores.to_parquet(proc / ANOMALY_SCORES_FILE)
    return scores
