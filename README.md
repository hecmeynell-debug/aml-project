# Anti-money laundering detection on the IBM synthetic transactions dataset

An end-to-end analytics project on the IBM "Transactions for Anti Money Laundering" data
(Altman et al., NeurIPS 2023; Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`).
It covers ingestion and currency normalisation, graph analysis, rule-based typology detectors,
unsupervised account anomaly detection, a supervised transaction model with strictly temporal
evaluation, and an interactive investigation dashboard.

> **Status:** Phases 0-3 are complete and documented below. Results for Phases 4-6 are filled in
> once the full pipeline has run (see "Headline results").

## Set-up

Requires Python 3.10+ (developed on 3.13, Windows 11, 16 GB RAM).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -e .   # Linux/macOS: .venv/bin/python
```

Data: set `KAGGLE_API_TOKEN` (or save `~/.kaggle/access_token`), or place
`HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/raw/` manually. Never commit credentials.

`hdbscan` is skipped on Python 3.13 (no wheel); `sklearn.cluster.HDBSCAN` is used instead.

## Commands

`make` is not installed on Windows by default, so every target is also available as
`python -m aml.cli <target>`; the Makefile is a thin wrapper.

| Target | What it does |
|---|---|
| `make data` | Download, ingest to parquet, derive FX rates and `amt_usd`, parse and match the patterns file |
| `make features` | Account-level features and leakage-free transaction features |
| `make detect` | Rule-based detectors, detector evaluation, anomaly models and their evaluation |
| `make train` | Supervised LightGBM model, evaluation, SHAP plot, and precomputed score tables |
| `make app` | Launch the Streamlit dashboard (reads precomputed scores only) |
| `make test` | Run the test suite |

Everything is configured in `config.yaml`. Switching to `HI-Medium` or `LI-Small` needs only
`dataset.name` to change. Notebooks `01_eda`, `02_graph` and `03_results` are generated from the
`.py` files beside them with `jupytext` and executed with `jupyter nbconvert`.

## Methods

**Ingestion.** DuckDB reads the CSV (the header repeats `Account`, so columns are named by
position). Bank codes are kept as strings because leading zeros matter (`010` is not `10`).
Account IDs are only unique within a bank, so nodes are `f"{bank}_{account}"`. `txn_id` is the
row position in the raw file.

**Currency.** The data has no FX table. Rates are the median `amount_received / amount_paid`
per currency pair over cross-currency transactions of at least 100 units (tiny amounts are
rounded and noisy), chained to USD by breadth-first search from USD. `amt_usd` is the amount
paid times the paying currency's USD rate. The table is saved to `reports/tables/fx_rates.csv`
and sanity-checked against loose plausible ranges.

**Graph.** Whole-graph measures use DuckDB and scipy; NetworkX is used only on subgraphs.
`ego_subgraph` extracts a time-filtered k-hop neighbourhood straight from parquet and prunes to
the highest-value links beyond `max_nodes`.

**Detectors** (`src/aml/detectors.py`; each returns `detector, account_ids, txn_ids, start_ts,
end_ts, total_usd, score, detail`):

- *Temporal cycles.* Restricted to non-trivial strongly connected components. A time-respecting
  depth-first search finds loops of at most `cycle_length_bound` hops in which every hop is
  strictly later than the previous one, the loop fits within `cycle_window_days`, and every hop's
  USD amount is within `amount_tolerance` of the first. This accepts exactly the cycles an
  "enumerate simple cycles, then verify timing" pipeline would, without enumerating the many that
  fail the time test. Work and cycle caps are logged when hit.
- *Fan-in / fan-out.* Peak distinct counterparties in sliding 1- and 7-day windows (DuckDB window
  functions), flagged strictly above a population percentile.
- *Scatter-gather / gather-scatter.* Time-ordered two-hop paths. Scatter-gather: source and
  destination joined by at least `min_paths` distinct intermediaries. Gather-scatter: a hub that
  collects from at least `min_paths` sources and redistributes to at least `min_paths`
  destinations.
- *Pass-through.* Outflow close to inflow, short median dwell, enough activity. **This is a
  behavioural proxy for shell accounts.** The data has no jurisdiction or offshore field, so shell
  companies cannot be identified, only conduit-like behaviour.

**Account anomaly detection.** About 30 account features (degrees, flows, format mix, bursts,
dwell, PageRank, clustering, detector hit counts), log-transformed where heavy-tailed and
standardised. Isolation Forest on the full matrix; LOF and HDBSCAN in a 10-component PCA
subspace. HDBSCAN flags noise and small clusters. The ensemble is the mean normalised rank.
Metrics are tie-aware (many HDBSCAN scores tie at "noise").

**Supervised model.** LightGBM at transaction level, `scale_pos_weight`, early stopping on
validation PR-AUC, decision threshold tuned for minority-class F1 on validation. Split by time
(60/20/20 by transaction order). Features use only transactions *strictly earlier* than the one
scored: for sender and receiver, prior count, USD volume, distinct counterparties and
cross-currency share in 1/7/30-day windows, plus own-transaction features and whether the pair
has transacted before. Phase 3 findings and Phase 4 account features are **not** used, since they
are computed over the whole period and would leak information from after the split boundaries.
The `is_laundering` label is used only as the training target and for evaluation.

## Headline results

*(Phases 1-3 measured on HI-Small; Phases 4-6 to be completed.)*

- 5,078,345 transactions, 515,088 accounts, 17.7 days, 5,177 laundering transactions (0.102%).
- Only 3,209 of the 5,177 laundering transactions (62%) belong to a listed typology attempt.
- Laundering is 5x over-represented inside non-trivial strongly connected components (23.5% of
  laundering vs 4.9% of all transactions).
- The union of rule-based detectors covers 69% of the 370 laundering attempts, at 0.23%
  transaction-level precision. Individual detectors range from 0.13% (pass-through) to 33.9%
  (scatter-gather). The cycle detector finds all cycles of up to 3 hops, 82% of 4-6 hops and none
  of the longer ones (bound of 6).

## Limitations

- **Synthetic data embeds its designers' assumptions.** The generator injects a fixed set of
  typologies with regular structure. Detectors and models that do well here may be learning the
  generator, not money laundering. Real laundering is adaptive, messier and rarer.
- **A time artefact dominates the late period.** Licit activity almost stops after 10 September
  while laundering attempts run on, so from 11 September 58-73% of transactions are laundering.
  Any model can exploit "little activity around" as a signal. The temporal test split straddles
  this regime; metrics are reported overall and by period for that reason.
- **No jurisdiction data.** There is no offshore or country attribute, so "shell-like" accounts
  are only a behavioural proxy (pass-through), not a statement about corporate structure.
- **Labels are incomplete for typologies.** 38% of laundering transactions have no attempt
  attribution, which caps typology-level recall and makes recall by typology a lower bound.
- **A real bank sees only its own slice of the network.** This project sees every account at
  every bank, including counterparties' onward flows. Cross-institution detection like this is
  not available to a single institution without data sharing.
- **False positives are costly.** At a base rate near 0.1%, even a detector with 99% specificity
  produces mostly false alerts. Each alert costs analyst time and can delay legitimate customers.
  Precision, not recall alone, should drive thresholds, and scores here are for triage, not
  decisions.
- **Scores on training data are optimistic.** The dashboard labels each transaction's split; only
  the test split is a fair evaluation.
- Cycle search is bounded by length (default 6, so longer laundering cycles are missed) and by
  work caps; gather-scatter hubs above `max_intermediary_txns` are skipped.

## Repository layout

```
config.yaml            all settings
src/aml/               ingest, currency, patterns, graph, detectors, features, anomaly,
                       supervised, evaluate, scoring, cli
app/streamlit_app.py   dashboard
notebooks/             01_eda, 02_graph, 03_results (+ .py sources)
reports/               tables/ (tracked) and figures/ (git-ignored, regenerated)
tests/                 pytest suite (also covers the dashboard with AppTest)
```
