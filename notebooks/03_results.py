# %% [markdown]
# # 03 · Consolidated results
# Every table and figure here is read from `reports/` (written by the pipeline); nothing is
# recomputed. Run `python -m aml.cli data features detect train` first.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

from aml.config import load_config

cfg = load_config()
TAB, FIG = cfg.root / cfg.paths.tables_dir, cfg.root / cfg.paths.figures_dir
pd.options.display.float_format = "{:,.4f}".format

def table(name, **kw):
    path = TAB / name
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame({"missing": [name]})

def figure(name, width=9):
    path = FIG / name
    if not path.exists():
        print(f"missing figure: {name}"); return
    img = mpimg.imread(path)
    fig, ax = plt.subplots(figsize=(width, width * img.shape[0] / img.shape[1]))
    ax.imshow(img); ax.axis("off"); plt.show()

# %% [markdown]
# ## 1. Data
# %%
display(table("eda_scale.csv")); display(table("eda_class_balance.csv"))
display(table("eda_rate_by_format.csv")); display(table("fx_rates.csv"))
figure("eda_activity_over_time.png")

# %% [markdown]
# ## 2. Graph structure
# %%
display(table("graph_components.csv", index_col=0)); display(table("graph_scc_share.csv", index_col=0))
display(table("graph_illicit_context.csv")); display(table("graph_ego_timing.csv"))
figure("graph_degree_distributions.png")

# %% [markdown]
# ## 3. Rule-based detectors
# Recall: share of laundering attempts with at least half their transactions covered.
# Diagonal cells matter; small attempts are covered by coincidence (see attempt-size table).
# %%
display(table("detector_recall_by_typology.csv", index_col=0))
display(table("detector_precision.csv", index_col=0))
display(table("detector_recall_by_attempt_size.csv")); display(table("detector_cycle_recall_by_hops.csv"))

# %% [markdown]
# ## 4. Unsupervised account anomaly detection
# %%
display(table("anomaly_results.csv", index_col=0)); figure("anomaly_pr_curves.png", 7)

# %% [markdown]
# ## 5. Supervised transaction model (temporal split)
# The test window straddles the late period in which licit activity collapses, so the
# `test_before_11sep` / `test_from_11sep` rows separate the two regimes.
# %%
display(table("supervised_metrics.csv", index_col=0))
display(table("supervised_recall_by_typology.csv", index_col=0))
display(table("supervised_feature_importance.csv", index_col=0).head(15))
figure("shap_summary.png", 8)
