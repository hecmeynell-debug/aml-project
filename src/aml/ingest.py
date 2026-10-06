"""Load the raw IBM AML transactions CSV, clean it and write it to parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from aml.config import Config

logger = logging.getLogger(__name__)

# The raw CSV repeats the header "Account" for both parties, so names are assigned positionally.
RAW_COLUMNS: list[str] = [
    "timestamp", "from_bank", "from_acct", "to_bank", "to_acct",
    "amt_received", "recv_ccy", "amt_paid", "pay_ccy", "payment_format", "is_laundering",
]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"


def make_node_id(bank: str | pd.Series, acct: str | pd.Series) -> str | pd.Series:
    """Global node ID. Account IDs are only unique within a bank, so prefix the bank.

    Bank codes are kept as strings: they carry leading zeros (``010`` and ``10`` differ).
    """
    return bank + "_" + acct


def ingest(cfg: Config, csv_path: Path | None = None, out_path: Path | None = None) -> Path:
    """Read the raw CSV with DuckDB, add node IDs, ``txn_id`` and ``is_self_loop``, write parquet.

    ``txn_id`` is the 0-based row position in the raw file, so it is stable across runs.
    """
    csv_path = csv_path or cfg.raw_trans_path
    out_path = out_path or cfg.transactions_parquet
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m aml.cli data` first.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = ", ".join(f"'{c}'" for c in RAW_COLUMNS)
    query = f"""
    COPY (
        WITH raw AS (
            SELECT row_number() OVER () - 1 AS txn_id, *
            FROM read_csv('{csv_path.as_posix()}', header = true, names = [{names}],
                          all_varchar = true)
        )
        SELECT
            CAST(txn_id AS BIGINT) AS txn_id,
            strptime(timestamp, '{TIMESTAMP_FORMAT}') AS timestamp,
            from_bank, from_acct, to_bank, to_acct,
            from_bank || '_' || from_acct AS src,
            to_bank || '_' || to_acct AS dst,
            CAST(amt_received AS DOUBLE) AS amt_received, recv_ccy,
            CAST(amt_paid AS DOUBLE) AS amt_paid, pay_ccy,
            payment_format,
            CAST(is_laundering AS TINYINT) AS is_laundering,
            (from_bank = to_bank AND from_acct = to_acct) AS is_self_loop
        FROM raw
        ORDER BY txn_id
    ) TO '{out_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """
    logger.info("Ingesting %s", csv_path)
    duckdb.sql(query)
    summary = duckdb.sql(
        f"""SELECT count(*) AS n, count(DISTINCT src) + 0 AS n_src,
                   sum(is_laundering) AS n_illicit, sum(is_self_loop::INT) AS n_self_loops,
                   min(timestamp) AS t0, max(timestamp) AS t1
            FROM read_parquet('{out_path.as_posix()}')"""
    ).fetchone()
    logger.info(
        "Wrote %s: %s rows, %s sending accounts, %s illicit, %s self-loops, %s to %s",
        out_path, *summary,
    )
    return out_path


def load_transactions(cfg: Config, columns: list[str] | None = None) -> pd.DataFrame:
    """Load the processed parquet into pandas (optionally a subset of columns)."""
    path = cfg.transactions_parquet
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run ingestion first.")
    return pd.read_parquet(path, columns=columns)
