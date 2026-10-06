"""Command-line entry points behind the ``make`` targets.

Usage: ``python -m aml.cli <data|features|detect|train|app|test>``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from aml.config import Config, load_config

logger = logging.getLogger("aml")

MANUAL_INSTRUCTIONS = """\
Kaggle credentials were not found. Either:
  1. Create an API token at https://www.kaggle.com/settings (API section) and either
     set the KAGGLE_API_TOKEN environment variable, or save the token to
     ~/.kaggle/access_token (or kaggle.json for legacy keys); or
  2. Download the dataset manually from
     https://www.kaggle.com/datasets/{slug}
     and place these files in {raw_dir}:
       - {trans}
       - {patterns}
"""


def _has_kaggle_credentials() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    return any(
        (
            os.environ.get("KAGGLE_API_TOKEN"),
            os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"),
            (kaggle_dir / "access_token").exists(),
            (kaggle_dir / "kaggle.json").exists(),
        )
    )


def download_data(cfg: Config) -> None:
    """Download the configured dataset's two files into the raw data directory."""
    raw_dir = cfg.root / cfg.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    wanted = [cfg.dataset.trans_filename, cfg.dataset.patterns_filename]
    if all((raw_dir / name).exists() for name in wanted):
        logger.info("Raw files already present in %s", raw_dir)
        return
    if not _has_kaggle_credentials():
        logger.error(
            MANUAL_INSTRUCTIONS.format(
                slug=cfg.dataset.kaggle_slug,
                raw_dir=raw_dir,
                trans=wanted[0],
                patterns=wanted[1],
            )
        )
        raise SystemExit(1)
    kaggle = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    for name in wanted:
        if (raw_dir / name).exists():
            continue
        logger.info("Downloading %s", name)
        subprocess.run(
            [kaggle, "datasets", "download", "-d", cfg.dataset.kaggle_slug,
             "-f", name, "-p", str(raw_dir)],
            check=True,
        )
        zip_path = raw_dir / f"{name}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(raw_dir)
            zip_path.unlink()
    for name in wanted:
        if not (raw_dir / name).exists():
            raise SystemExit(f"Expected file missing after download: {raw_dir / name}")
    logger.info("Download complete")


def run_tests(_: Config) -> None:
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))


def run_app(cfg: Config) -> None:
    app = cfg.root / "app" / "streamlit_app.py"
    if not (cfg.root / cfg.paths.processed_dir / "account_scores.parquet").exists():
        raise SystemExit("Score tables missing. Run `python -m aml.cli train` first.")
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="aml", description=__doc__)
    parser.add_argument("command", choices=["data", "features", "detect", "train", "app", "test"])
    parser.add_argument("--config", default=None, help="Path to an alternative config.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.command == "data":
        from aml.currency import add_usd_amounts
        from aml.ingest import ingest
        from aml.patterns import build_patterns

        download_data(cfg)
        ingest(cfg)
        add_usd_amounts(cfg)
        build_patterns(cfg)
    elif args.command == "features":
        from aml.features import build_account_features, build_transaction_features

        build_account_features(cfg)
        build_transaction_features(cfg)
    elif args.command == "detect":
        from aml.anomaly import run_anomaly
        from aml.detectors import detect_all
        from aml.evaluate import evaluate_anomaly, evaluate_detectors

        findings = detect_all(cfg)
        evaluate_detectors(cfg, findings)
        evaluate_anomaly(cfg, run_anomaly(cfg, findings))
    elif args.command == "train":
        from aml.scoring import build_all_scores
        from aml.supervised import train_supervised

        train_supervised(cfg)
        build_all_scores(cfg)  # precomputed tables the dashboard reads
    elif args.command == "app":
        run_app(cfg)
    elif args.command == "test":
        run_tests(cfg)


if __name__ == "__main__":
    main()
