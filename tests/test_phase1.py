"""Tests for node IDs, currency normalisation and the patterns parser."""

from pathlib import Path

import pandas as pd
import pytest

from aml.currency import chain_to_usd, check_rates, normalise_to_usd
from aml.ingest import make_node_id
from aml.patterns import KEY_COLUMNS, match_to_transactions, parse_patterns

FIXTURE = Path(__file__).parent / "fixtures" / "mini_patterns.txt"


def test_node_id_keeps_leading_zeros_and_bank_scope() -> None:
    a = make_node_id("010", "8000EBD30")
    b = make_node_id("10", "8000EBD30")
    assert a == "010_8000EBD30"
    assert a != b  # same account number at different banks is a different node
    s = make_node_id(pd.Series(["01", "02"]), pd.Series(["A", "A"]))
    assert list(s) == ["01_A", "02_A"]


def test_chain_to_usd_and_normalisation_on_toy_frame() -> None:
    # 1 USD = 0.5 Euro-units?? no: 1 Euro = 1.25 USD  =>  USD->Euro rate 0.8
    pairs = pd.DataFrame(
        {
            "pay_ccy": ["US Dollar", "Euro"],
            "recv_ccy": ["Euro", "Yen"],
            "n": [100, 80],
            "rate": [0.8, 100.0],  # 1 USD = 0.8 EUR; 1 EUR = 100 JPY
        }
    )
    rates = chain_to_usd(pairs).set_index("currency")["usd_per_unit"]
    assert rates["US Dollar"] == pytest.approx(1.0)
    assert rates["Euro"] == pytest.approx(1.25)
    assert rates["Yen"] == pytest.approx(0.0125)
    df = pd.DataFrame({"amt_paid": [100.0, 1000.0, 5.0], "pay_ccy": ["Euro", "Yen", "US Dollar"]})
    out = normalise_to_usd(df, rates.reset_index())
    assert out["amt_usd"].tolist() == pytest.approx([125.0, 12.5, 5.0])


def test_normalise_raises_on_unknown_currency() -> None:
    rates = pd.DataFrame({"currency": ["US Dollar"], "usd_per_unit": [1.0]})
    with pytest.raises(ValueError):
        normalise_to_usd(pd.DataFrame({"amt_paid": [1.0], "pay_ccy": ["Gold"]}), rates)


def test_check_rates_warns_on_implausible_value() -> None:
    rates = pd.DataFrame({"currency": ["US Dollar", "Euro"], "usd_per_unit": [1.0, 50.0]})
    pairs = pd.DataFrame(columns=["pay_ccy", "recv_ccy", "n", "rate"])
    warnings = check_rates(rates, pairs)
    assert any("Euro" in w and "outside plausible" in w for w in warnings)


def test_parse_patterns_fixture() -> None:
    df = parse_patterns(FIXTURE)
    assert len(df) == 3
    assert df["attempt_id"].tolist() == [0, 0, 1]
    assert df["typology"].tolist() == ["FAN-OUT", "FAN-OUT", "BIPARTITE"]
    assert df["detail"].tolist()[0] == "Max 2-degree Fan-Out"
    assert df["detail"].tolist()[2] == ""
    assert df["from_bank"].iloc[0] == "021174"  # leading zeros preserved
    assert df["amt_paid"].iloc[1] == pytest.approx(8630.40)
    assert df["timestamp"].iloc[0] == pd.Timestamp("2022-09-01 00:06")


def test_parse_patterns_rejects_unterminated_attempt(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("BEGIN LAUNDERING ATTEMPT - CYCLE: Max 2 hops\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_patterns(bad)


def test_match_to_transactions_pairs_duplicates_in_order() -> None:
    patterns = parse_patterns(FIXTURE)
    base = patterns.iloc[[0]][KEY_COLUMNS].copy()
    # two identical illicit transactions plus one licit decoy with the same key
    txns = pd.concat([base, base, base], ignore_index=True)
    txns["txn_id"] = [10, 11, 12]
    txns["is_laundering"] = [1, 1, 0]
    two_same = pd.concat([patterns.iloc[[0]], patterns.iloc[[0]]], ignore_index=True)
    two_same["attempt_id"] = [0, 1]
    two_same["seq"] = [0, 0]
    matched = match_to_transactions(two_same, txns)
    assert sorted(matched["txn_id"].tolist()) == [10, 11]
    # an unmatched row yields NaN rather than raising
    unmatched = match_to_transactions(patterns.iloc[[2]], txns)
    assert unmatched["txn_id"].isna().all()
