# %% [markdown]
# # 01 · Exploratory data analysis (HI-Small)
# Scale, class balance, laundering rates, amounts, time profile and typologies.
# Figures are written to `reports/figures/`. Aggregation is done in DuckDB.

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aml.config import load_config

cfg = load_config()
FIG = cfg.root / cfg.paths.figures_dir
TAB = cfg.root / cfg.paths.tables_dir
P = cfg.transactions_parquet.as_posix()
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
LICIT, ILLICIT = "#4C72B0", "#C44E52"
q = lambda sql: duckdb.sql(sql).df()

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Scale of the data

# %%
scale = q(f"""
    SELECT (SELECT count(*) FROM '{P}') AS transactions,
           (SELECT count(*) FROM (SELECT src FROM '{P}' UNION SELECT dst FROM '{P}')) AS accounts,
           (SELECT count(*) FROM (SELECT from_bank FROM '{P}' UNION SELECT to_bank FROM '{P}')) AS banks,
           (SELECT min(timestamp) FROM '{P}') AS first_ts, (SELECT max(timestamp) FROM '{P}') AS last_ts
""")
scale["span_days"] = (scale.last_ts - scale.first_ts).dt.total_seconds() / 86400
scale.to_csv(TAB / "eda_scale.csv", index=False)
scale.T

# %% [markdown]
# ## Class balance

# %%
bal = q(f"SELECT is_laundering, count(*) AS n FROM '{P}' GROUP BY 1 ORDER BY 1")
bal["share_pct"] = 100 * bal.n / bal.n.sum()
bal.to_csv(TAB / "eda_class_balance.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 3.2))
ax.bar(["Licit", "Laundering"], bal.n, color=[LICIT, ILLICIT])
ax.set_yscale("log"); ax.set_ylabel("Transactions (log)")
for i, r in bal.iterrows():
    ax.text(i, r.n, f"{int(r.n):,}\n({r.share_pct:.3f}%)", ha="center", va="bottom", fontsize=8)
ax.set_ylim(top=bal.n.max() * 8); ax.set_title("Class balance")
save(fig, "eda_class_balance")
bal

# %% [markdown]
# ## Laundering rate by payment format and by currency

# %%
by_fmt = q(f"""SELECT payment_format AS grp, count(*) n, sum(is_laundering) ill,
               100.0*avg(is_laundering) AS rate_pct FROM '{P}' GROUP BY 1 ORDER BY rate_pct DESC""")
by_ccy = q(f"""SELECT pay_ccy AS grp, count(*) n, sum(is_laundering) ill,
               100.0*avg(is_laundering) AS rate_pct FROM '{P}' GROUP BY 1 ORDER BY rate_pct DESC""")
by_fmt.to_csv(TAB / "eda_rate_by_format.csv", index=False)
by_ccy.to_csv(TAB / "eda_rate_by_currency.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, d, t in zip(axes, (by_fmt, by_ccy), ("payment format", "paying currency")):
    ax.barh(d.grp[::-1], d.rate_pct[::-1], color=ILLICIT)
    ax.set_xlabel("Laundering rate (%)"); ax.set_title(f"Laundering rate by {t}")
save(fig, "eda_rate_by_format_currency")
display(by_fmt); display(by_ccy)

# %% [markdown]
# ## Cross-currency share, licit vs illicit
# Note: cross-currency transactions are rare and, in HI-Small, never laundering.

# %%
xc = q(f"""SELECT is_laundering, count(*) n, sum((pay_ccy<>recv_ccy)::INT) xc,
           100.0*avg((pay_ccy<>recv_ccy)::INT) AS xc_pct FROM '{P}' GROUP BY 1 ORDER BY 1""")
xc.to_csv(TAB / "eda_cross_currency.csv", index=False)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.bar(["Licit", "Laundering"], xc.xc_pct, color=[LICIT, ILLICIT])
ax.set_ylabel("Cross-currency share (%)"); ax.set_title("Cross-currency transactions")
save(fig, "eda_cross_currency")
xc

# %% [markdown]
# ## Amount distributions (USD, log scale)

# %%
amt = q(f"""SELECT is_laundering, amt_usd FROM '{P}' WHERE amt_usd > 0
            AND (is_laundering = 1 OR random() < 0.05)""")
bins = np.logspace(-2, 11, 80)
fig, ax = plt.subplots(figsize=(7, 3.8))
for lab, col, name in ((0, LICIT, "Licit (5% sample)"), (1, ILLICIT, "Laundering")):
    ax.hist(amt.loc[amt.is_laundering == lab, "amt_usd"], bins=bins, density=True,
            alpha=0.6, color=col, label=name)
ax.set_xscale("log"); ax.set_xlabel("Amount (USD)"); ax.set_ylabel("Density")
ax.legend(); ax.set_title("Transaction amount distribution")
save(fig, "eda_amount_distribution")
amt.groupby("is_laundering").amt_usd.describe(percentiles=[.25, .5, .75, .99])

# %% [markdown]
# ## Round amounts
# A round amount here is a multiple of 100 in the paying currency.

# %%
rnd = q(f"""SELECT is_laundering, 100.0*avg((amt_paid % 100 = 0)::INT) AS round_pct,
            100.0*avg((amt_paid % 1 = 0)::INT) AS whole_pct FROM '{P}' GROUP BY 1 ORDER BY 1""")
rnd.to_csv(TAB / "eda_round_amounts.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 3.2))
x = np.arange(2)
ax.bar(x - 0.2, rnd.round_pct, 0.4, label="Multiple of 100", color=LICIT)
ax.bar(x + 0.2, rnd.whole_pct, 0.4, label="Whole number", color=ILLICIT)
ax.set_xticks(x, ["Licit", "Laundering"]); ax.set_ylabel("Share of transactions (%)")
ax.legend(); ax.set_title("Round amounts")
save(fig, "eda_round_amounts")
rnd

# %% [markdown]
# ## Activity and laundering rate over time

# %%
hourly = q(f"""SELECT date_trunc('hour', timestamp) AS h, count(*) n FROM '{P}' GROUP BY 1 ORDER BY 1""")
daily = q(f"""SELECT date_trunc('day', timestamp) AS d, count(*) n, sum(is_laundering) ill,
              100.0*avg(is_laundering) AS rate_pct FROM '{P}' GROUP BY 1 ORDER BY 1""")
daily.to_csv(TAB / "eda_daily.csv", index=False)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(hourly.h, hourly.n, color=LICIT, lw=0.9); axes[0].set_ylabel("Transactions / hour")
axes[0].set_title("Activity over time")
axes[1].bar(daily.d, daily.rate_pct, width=0.8, color=ILLICIT, align="edge")
axes[1].set_ylabel("Laundering rate (%) / day"); axes[1].set_title("Laundering rate over time")
save(fig, "eda_activity_over_time")
daily

# %% [markdown]
# ## Self-loops

# %%
sl = q(f"""SELECT is_laundering, sum(is_self_loop::INT) AS self_loops, count(*) n,
           sum(is_self_loop::INT) FILTER (WHERE payment_format='Reinvestment') AS reinvestment
           FROM '{P}' GROUP BY 1 ORDER BY 1""")
sl.to_csv(TAB / "eda_self_loops.csv", index=False)
sl

# %% [markdown]
# ## Typologies in the patterns file

# %%
pat = pd.read_parquet(cfg.root / cfg.paths.processed_dir / "patterns.parquet")
typ = pat.groupby("typology").agg(attempts=("attempt_id", "nunique"), transactions=("seq", "size"))
typ["txns_per_attempt"] = typ.transactions / typ.attempts
typ = typ.sort_values("transactions", ascending=False)
typ.to_csv(TAB / "eda_typologies.csv")
n_ill = int(bal.loc[bal.is_laundering == 1, "n"].iloc[0])
print(f"Matched {pat.txn_id.notna().sum()}/{len(pat)} pattern rows; "
      f"{pat.txn_id.nunique()} of {n_ill} laundering txns ({100*pat.txn_id.nunique()/n_ill:.1f}%) are in a listed attempt")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].barh(typ.index[::-1], typ.attempts[::-1], color=LICIT); axes[0].set_title("Attempts per typology")
axes[1].barh(typ.index[::-1], typ.transactions[::-1], color=ILLICIT); axes[1].set_title("Transactions per typology")
save(fig, "eda_typologies")
typ
