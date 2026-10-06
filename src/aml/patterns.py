"""Parse ``*_Patterns.txt`` (laundering attempts grouped by typology) and match rows to ``txn_id``.

File format (verified on HI-Small)::

    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
    ...
    END LAUNDERING ATTEMPT - FAN-OUT

Rows have the same 11 fields as the transactions CSV. Some headers carry no detail
(``BIPARTITE``, ``STACK``, ``SCATTER-GATHER``...).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from aml.config import Config
from aml.ingest import RAW_COLUMNS, TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - (?P<typology>[A-Z-]+)(?::\s*(?P<detail>.*))?\s*$")
_END = re.compile(r"^END LAUNDERING ATTEMPT - (?P<typology>[A-Z-]+)\s*$")

KEY_COLUMNS: list[str] = [
    "timestamp", "from_bank", "from_acct", "to_bank", "to_acct",
    "amt_received", "recv_ccy", "amt_paid", "pay_ccy", "payment_format",
]
PATTERN_COLUMNS: list[str] = ["attempt_id", "typology", "detail", "seq", *RAW_COLUMNS]


def parse_patterns(path: Path) -> pd.DataFrame:
    """Parse a patterns file into one row per transaction listed in an attempt.

    Columns: ``attempt_id`` (0-based, in file order), ``typology``, ``detail`` (free text from the
    header, possibly empty), ``seq`` (position within the attempt) plus the 11 transaction fields.
    """
    records: list[list[object]] = []
    attempt_id = -1
    typology: str | None = None
    detail = ""
    seq = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            begin = _BEGIN.match(line)
            if begin:
                if typology is not None:
                    raise ValueError(f"{path}:{lineno}: BEGIN inside an open attempt")
                attempt_id += 1
                typology = begin.group("typology")
                detail = (begin.group("detail") or "").strip()
                seq = 0
                continue
            end = _END.match(line)
            if end:
                if typology is None or end.group("typology") != typology:
                    raise ValueError(f"{path}:{lineno}: unmatched END line")
                typology = None
                continue
            if typology is None:
                raise ValueError(f"{path}:{lineno}: transaction row outside an attempt")
            fields = line.split(",")
            if len(fields) != len(RAW_COLUMNS):
                raise ValueError(f"{path}:{lineno}: expected {len(RAW_COLUMNS)} fields, got {len(fields)}")
            records.append([attempt_id, typology, detail, seq, *fields])
            seq += 1
    if typology is not None:
        raise ValueError(f"{path}: file ended inside an open attempt")
    df = pd.DataFrame(records, columns=PATTERN_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT)
    for col in ("amt_received", "amt_paid"):
        df[col] = df[col].astype(float)
    df["is_laundering"] = df["is_laundering"].astype("int8")
    return df


def match_to_transactions(patterns: pd.DataFrame, txns: pd.DataFrame) -> pd.DataFrame:
    """Add ``txn_id`` to ``patterns`` by matching on the 10 non-label fields (NaN if unmatched).

    Identical rows can occur several times in the transactions table. They are paired in file
    order: the n-th pattern row with a given key takes the n-th matching laundering transaction
    (cycling if the key occurs more often in the patterns file than in the transactions).
    """
    illicit = txns.loc[txns["is_laundering"] == 1, ["txn_id", *KEY_COLUMNS]].copy()
    illicit = illicit.sort_values("txn_id")
    illicit["_rank"] = illicit.groupby(KEY_COLUMNS).cumcount()
    illicit["_k"] = illicit.groupby(KEY_COLUMNS)["txn_id"].transform("size")

    out = patterns.copy()
    out["_occ"] = out.groupby(KEY_COLUMNS).cumcount()
    k = out[KEY_COLUMNS].merge(
        illicit.drop_duplicates(KEY_COLUMNS)[[*KEY_COLUMNS, "_k"]], on=KEY_COLUMNS, how="left"
    )["_k"]
    out["_rank"] = (out["_occ"].to_numpy() % k.to_numpy()).astype("float")
    merged = out.merge(
        illicit[[*KEY_COLUMNS, "_rank", "txn_id"]], on=[*KEY_COLUMNS, "_rank"], how="left"
    )
    merged = merged.drop(columns=["_occ", "_rank"]).sort_values(["attempt_id", "seq"])
    return merged.reset_index(drop=True)


def build_patterns(cfg: Config) -> pd.DataFrame:
    """Parse the configured patterns file, match to transactions, save parquet and log the rate."""
    patterns = parse_patterns(cfg.raw_patterns_path)
    txns = pd.read_parquet(cfg.transactions_parquet, columns=["txn_id", "is_laundering", *KEY_COLUMNS])
    matched = match_to_transactions(patterns, txns)
    n, hit = len(matched), int(matched["txn_id"].notna().sum())
    logger.info(
        "Patterns: %d attempts, %d rows; matched %d/%d (%.2f%%) to txn_id; %d distinct txns",
        matched["attempt_id"].nunique(), n, hit, n, 100 * hit / n, matched["txn_id"].nunique(),
    )
    n_ill = int((txns["is_laundering"] == 1).sum())
    logger.info(
        "%d of %d laundering transactions (%.1f%%) belong to a listed attempt",
        matched["txn_id"].nunique(), n_ill, 100 * matched["txn_id"].nunique() / n_ill,
    )
    out = cfg.root / cfg.paths.processed_dir / "patterns.parquet"
    matched.to_parquet(out, index=False)
    return matched
