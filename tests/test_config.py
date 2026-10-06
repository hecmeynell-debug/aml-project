"""Smoke tests for the configuration loader."""

from pathlib import Path

import pytest

from aml.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.dataset.name == "HI-Small"
    assert cfg.seed == 42
    assert cfg.detectors.fan_windows_days == (1, 7)
    assert cfg.raw_trans_path.name == "HI-Small_Trans.csv"
    assert cfg.raw_patterns_path.name == "HI-Small_Patterns.txt"


def test_dataset_switch_needs_only_config(tmp_path: Path) -> None:
    alt = tmp_path / "alt.yaml"
    alt.write_text("dataset:\n  name: LI-Small\n", encoding="utf-8")
    cfg = load_config(alt)
    assert cfg.raw_trans_path.name == "LI-Small_Trans.csv"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("detectors:\n  not_a_setting: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
