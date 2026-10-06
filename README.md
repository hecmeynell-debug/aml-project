# AML detection on the IBM synthetic transactions dataset

Work in progress (Phase 0 complete). Full documentation is written in Phase 7.

## Set-up

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -e .
```

Kaggle credentials: set `KAGGLE_API_TOKEN` (or save `~/.kaggle/access_token`), or place
`HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/raw/` manually.

## Commands

`make <target>` or, where `make` is unavailable, `python -m aml.cli <target>`:
`data`, `features`, `detect`, `train`, `app`, `test`.

`hdbscan` is skipped on Python 3.13; `sklearn.cluster.HDBSCAN` is used instead.
