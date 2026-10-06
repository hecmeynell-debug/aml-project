"""Typed configuration loaded from ``config.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "HI-Small"
    kaggle_slug: str = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"

    @property
    def trans_filename(self) -> str:
        return f"{self.name}_Trans.csv"

    @property
    def patterns_filename(self) -> str:
        return f"{self.name}_Patterns.txt"


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    figures_dir: Path = Path("reports/figures")
    tables_dir: Path = Path("reports/tables")


@dataclass(frozen=True)
class GraphConfig:
    ego_max_nodes: int = 300


@dataclass(frozen=True)
class DetectorsConfig:
    cycle_length_bound: int = 6
    cycle_window_days: float = 14
    cycle_min_length: int = 2
    amount_tolerance: float = 0.20
    max_cycles_per_component: int = 100_000
    fan_windows_days: tuple[int, ...] = (1, 7)
    fan_percentile: float = 99.5
    scatter_gather_min_paths: int = 3
    scatter_gather_window_days: float = 7
    attempt_detection_threshold: float = 0.5
    max_expansions_per_component: int = 20_000_000
    fan_min_degree: int = 3
    max_intermediary_txns: int = 2000
    pass_through_ratio_tol: float = 0.10
    pass_through_max_dwell_hours: float = 24.0
    pass_through_min_txns: int = 2
    pass_through_min_usd: float = 1000.0
    max_txns_per_finding: int = 500


@dataclass(frozen=True)
class AnomalyConfig:
    min_cluster_size: int = 50
    contamination: float = 0.01
    pca_components: int = 10
    lof_neighbors: int = 20
    iforest_estimators: int = 200
    small_cluster_size: int = 200
    precision_at_k: tuple[int, ...] = (100, 500, 1000)


@dataclass(frozen=True)
class SupervisedConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    lookback_days: tuple[int, ...] = (1, 7, 30)
    pos_weight_power: float = 0.25


@dataclass(frozen=True)
class Config:
    """Top-level project configuration. Relative paths resolve against the project root."""

    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    detectors: DetectorsConfig = field(default_factory=DetectorsConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    supervised: SupervisedConfig = field(default_factory=SupervisedConfig)

    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    @property
    def raw_trans_path(self) -> Path:
        return self.root / self.paths.raw_dir / self.dataset.trans_filename

    @property
    def raw_patterns_path(self) -> Path:
        return self.root / self.paths.raw_dir / self.dataset.patterns_filename

    @property
    def transactions_parquet(self) -> Path:
        return self.root / self.paths.processed_dir / "transactions.parquet"


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a mapping, coercing lists/paths to the declared types."""
    data = data or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, list):
            value = tuple(value)
        if cls is PathsConfig:
            value = Path(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load ``config.yaml`` (or ``path``) into a :class:`Config`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    sections = {
        "dataset": DatasetConfig,
        "paths": PathsConfig,
        "graph": GraphConfig,
        "detectors": DetectorsConfig,
        "anomaly": AnomalyConfig,
        "supervised": SupervisedConfig,
    }
    unknown = set(raw) - set(sections) - {"seed"}
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {name: _build(cls, raw.get(name)) for name, cls in sections.items()}
    if "seed" in raw:
        kwargs["seed"] = int(raw["seed"])
    return Config(**kwargs)
