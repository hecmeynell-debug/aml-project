"""Implied FX rates and USD normalisation.

The dataset has no exchange-rate table. Rates are inferred from the (few) cross-currency
transactions, where ``amt_received / amt_paid`` is the number of receiving-currency units per
unit of paying currency.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import duckdb
import pandas as pd

from aml.config import Config

logger = logging.getLogger(__name__)

BASE_CCY = "US Dollar"
# Only transactions at least this large (in the paying currency) are used to estimate rates:
# amounts are rounded to 2 decimals, so tiny payments give very noisy ratios.
MIN_AMOUNT_FOR_RATE = 100.0
# Loose plausibility bounds on USD per one unit of each currency (any 2022 value should fit).
PLAUSIBLE_USD_PER_UNIT: dict[str, tuple[float, float]] = {
    "Euro": (0.8, 1.4),
    "UK Pound": (1.0, 1.6),
    "Swiss Franc": (0.85, 1.3),
    "Yuan": (0.1, 0.2),
    "Shekel": (0.2, 0.4),
    "Rupee": (0.008, 0.02),
    "Ruble": (0.005, 0.03),
    "Yen": (0.005, 0.015),
    "Canadian Dollar": (0.6, 0.9),
    "Australian Dollar": (0.55, 0.85),
    "Mexican Peso": (0.03, 0.07),
    "Saudi Riyal": (0.2, 0.3),
    "Brazil Real": (0.12, 0.3),
    "Bitcoin": (5_000.0, 100_000.0),
}


def pairwise_rates(parquet: Path, min_amount: float = MIN_AMOUNT_FOR_RATE) -> pd.DataFrame:
    """Median ``amt_received / amt_paid`` and count for every (pay_ccy, recv_ccy) pair."""
    return duckdb.sql(
        f"""
        SELECT pay_ccy, recv_ccy, count(*) AS n, median(amt_received / amt_paid) AS rate
        FROM read_parquet('{parquet.as_posix()}')
        WHERE pay_ccy <> recv_ccy AND amt_paid >= {min_amount}
        GROUP BY 1, 2
        """
    ).df()


def chain_to_usd(pairs: pd.DataFrame, base: str = BASE_CCY) -> pd.DataFrame:
    """Chain pairwise rates into USD-per-unit for every currency (breadth-first from USD).

    Edges are explored in order of descending sample size, so each currency is anchored on the
    best-supported route. Returns columns ``currency, usd_per_unit, via, n_obs, hops``.
    """
    # Each pair says: 1 pay_ccy = rate recv_ccy. Store both directions.
    adj: dict[str, list[tuple[str, float, int]]] = {}
    for row in pairs.sort_values("n", ascending=False).itertuples():
        adj.setdefault(row.pay_ccy, []).append((row.recv_ccy, row.rate, row.n))      # pay -> recv
        adj.setdefault(row.recv_ccy, []).append((row.pay_ccy, 1.0 / row.rate, row.n))  # recv -> pay
    usd: dict[str, tuple[float, str, int, int]] = {base: (1.0, base, 0, 0)}
    queue: deque[str] = deque([base])
    while queue:
        cur = queue.popleft()
        cur_usd, _, _, hops = usd[cur]
        for nxt, r, n in adj.get(cur, []):
            # 1 nxt = r' cur  where r' = 1 / r(cur->nxt)  =>  usd_per_nxt = usd_per_cur / r
            if nxt not in usd:
                usd[nxt] = (cur_usd / r, cur, n, hops + 1)
                queue.append(nxt)
    out = pd.DataFrame(
        [(c, v[0], v[1], v[2], v[3]) for c, v in usd.items()],
        columns=["currency", "usd_per_unit", "via", "n_obs", "hops"],
    )
    return out.sort_values("usd_per_unit", ascending=False).reset_index(drop=True)


def check_rates(rates: pd.DataFrame, pairs: pd.DataFrame, tol: float = 0.05) -> list[str]:
    """Return (and log) warnings: implausible rates, and pairs inconsistent with the chain."""
    warnings: list[str] = []
    usd = rates.set_index("currency")["usd_per_unit"]
    for ccy, (lo, hi) in PLAUSIBLE_USD_PER_UNIT.items():
        if ccy not in usd.index:
            warnings.append(f"No rate could be derived for {ccy}")
        elif not lo <= usd[ccy] <= hi:
            warnings.append(f"{ccy}: {usd[ccy]:.6g} USD/unit outside plausible range [{lo}, {hi}]")
    for row in pairs.itertuples():
        if row.pay_ccy in usd.index and row.recv_ccy in usd.index:
            chained = usd[row.pay_ccy] / usd[row.recv_ccy]
            if abs(row.rate / chained - 1) > tol and row.n >= 50:
                warnings.append(
                    f"{row.pay_ccy}->{row.recv_ccy}: direct {row.rate:.4g} vs chained "
                    f"{chained:.4g} (n={row.n})"
                )
    for w in warnings:
        logger.warning(w)
    return warnings


def normalise_to_usd(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Add ``amt_usd`` (amount paid x USD per unit of the paying currency) to ``df``."""
    mapping = rates.set_index("currency")["usd_per_unit"]
    factor = df["pay_ccy"].map(mapping)
    if factor.isna().any():
        missing = sorted(df.loc[factor.isna(), "pay_ccy"].unique())
        raise ValueError(f"No USD rate for currencies: {missing}")
    out = df.copy()
    out["amt_usd"] = out["amt_paid"] * factor
    return out


def add_usd_amounts(cfg: Config) -> pd.DataFrame:
    """Derive rates, save ``fx_rates.csv`` and add ``amt_usd`` to the processed parquet."""
    path = cfg.transactions_parquet
    pairs = pairwise_rates(path)
    logger.info("Derived %d currency-pair medians from cross-currency transactions", len(pairs))
    rates = chain_to_usd(pairs)
    logger.info("USD rate table:\n%s", rates.to_string(index=False))
    check_rates(rates, pairs)
    tables = cfg.root / cfg.paths.tables_dir
    tables.mkdir(parents=True, exist_ok=True)
    rates.to_csv(tables / "fx_rates.csv", index=False)
    pairs.sort_values("n", ascending=False).to_csv(tables / "fx_pair_medians.csv", index=False)

    df = pd.read_parquet(path)
    if "amt_usd" in df.columns:
        df = df.drop(columns="amt_usd")
    df = normalise_to_usd(df, rates)
    df.to_parquet(path, index=False, compression="zstd")
    logger.info("Added amt_usd; total volume %.3e USD", df["amt_usd"].sum())
    return rates
