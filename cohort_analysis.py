"""
cohort_analysis.py
------------------
Cohort retention analysis — groups customers by contract-start quarter,
tracks what % of each cohort remains active at 3, 6, 9, 12, 18, 24 months.
Outputs:
  • charts/cohort_retention_heatmap.png
  • charts/mrr_retention_by_cohort.png
  • data/cohort_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

os.makedirs("charts", exist_ok=True)
os.makedirs("data",   exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/customers.csv", parse_dates=["contract_start"])

REFERENCE_DATE = pd.Timestamp("2025-04-01")
CHECKPOINTS    = [3, 6, 9, 12, 18, 24]   # months post-signup

# ── Assign cohort (quarter of signup) ────────────────────────────────────────
df["cohort"] = df["contract_start"].dt.to_period("Q").astype(str)
df["tenure_months"] = (
    (REFERENCE_DATE - df["contract_start"]).dt.days / 30.44
).round(1)

# ── Build retention matrix ────────────────────────────────────────────────────
cohorts   = sorted(df["cohort"].unique())
ret_matrix = {}   # cohort → {checkpoint: retention %}
mrr_matrix = {}

for cohort in cohorts:
    cohort_df = df[df["cohort"] == cohort].copy()
    n_total   = len(cohort_df)
    mrr_total = cohort_df["mrr"].sum()
    if n_total < 5:
        continue

    ret_row = {}
    mrr_row = {}
    for cp in CHECKPOINTS:
        # Only include customers whose tenure reaches this checkpoint
        eligible = cohort_df[cohort_df["tenure_months"] >= cp]
        if len(eligible) == 0:
            ret_row[f"{cp}M"] = np.nan
            mrr_row[f"{cp}M"] = np.nan
            continue
        # Retained = not churned by that checkpoint
        retained     = eligible[eligible["churned"] == False]
        ret_row[f"{cp}M"] = round(len(retained) / len(eligible) * 100, 1)
        mrr_row[f"{cp}M"] = round(retained["mrr"].sum() / max(eligible["mrr"].sum(), 1) * 100, 1)

    ret_matrix[cohort] = ret_row
    mrr_matrix[cohort] = mrr_row

ret_df = pd.DataFrame(ret_matrix).T
mrr_df = pd.DataFrame(mrr_matrix).T

# Drop cohorts where we have no data at all
ret_df.dropna(how="all", inplace=True)
mrr_df.dropna(how="all", inplace=True)

ret_df.to_csv("data/cohort_summary.csv")

# ── Retention heatmap ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, max(6, len(ret_df) * 0.55)))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

# Custom colormap: dark red → amber → teal
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "cs_retention", ["#E05252", "#FFD166", "#4ABFA0"], N=256
)

# Mask NaN cells
mask = ret_df.isnull()
sns.heatmap(
    ret_df,
    ax=ax,
    cmap=cmap,
    annot=True,
    fmt=".0f",
    annot_kws={"size": 10, "weight": "bold", "color": "white"},
    linewidths=0.5,
    linecolor="#1C1F26",
    mask=mask,
    vmin=50, vmax=100,
    cbar_kws={"shrink": 0.6, "label": "Retention %"},
)

# Style annotations to be white
for text in ax.texts:
    text.set_color("white")

ax.set_title("Customer Cohort Retention by Quarter",
             color="white", fontsize=15, fontweight="bold", pad=18)
ax.set_xlabel("Months Since Contract Start", color="#AAAAAA", fontsize=11)
ax.set_ylabel("Signup Cohort (Quarter)",      color="#AAAAAA", fontsize=11)
ax.tick_params(colors="white", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color="white")
cbar.ax.tick_params(labelcolor="white")
cbar.set_label("Retention %", color="#AAAAAA")

plt.tight_layout()
plt.savefig("charts/cohort_retention_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("✅  Saved → charts/cohort_retention_heatmap.png")

# ── MRR retention line chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

COLORS = [
    "#4ABFA0", "#FFD166", "#E05252", "#7B9FE0", "#C77DFF",
    "#F4A261", "#2EC4B6", "#E9C46A", "#A8DADC", "#F1FAEE"
]

checkpoints_clean = [c for c in mrr_df.columns]
for idx, (cohort, row) in enumerate(mrr_df.iterrows()):
    valid = row.dropna()
    if len(valid) < 2:
        continue
    color = COLORS[idx % len(COLORS)]
    ax.plot(valid.index, valid.values,
            marker="o", linewidth=2, markersize=5,
            color=color, label=cohort, alpha=0.85)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylim(40, 105)
ax.set_xlabel("Months Since Contract Start", color="#AAAAAA", fontsize=11)
ax.set_ylabel("MRR Retained (%)",            color="#AAAAAA", fontsize=11)
ax.set_title("MRR Retention by Signup Cohort",
             color="white", fontsize=14, fontweight="bold", pad=16)

ax.tick_params(colors="white")
ax.grid(axis="y", color="#2A2D35", linewidth=0.8, linestyle="--")
ax.legend(title="Cohort", facecolor="#1C1F26", edgecolor="#333",
          labelcolor="white", title_fontsize=9, fontsize=8,
          loc="lower left", ncol=2)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("charts/mrr_retention_by_cohort.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("✅  Saved → charts/mrr_retention_by_cohort.png")

# ── Print key insights ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  COHORT RETENTION — KEY INSIGHTS")
print("=" * 55)
avg_6m  = ret_df["6M"].mean()
avg_12m = ret_df["12M"].mean()
avg_24m = ret_df["24M"].dropna().mean()
print(f"  Avg retention at  6 months : {avg_6m:.1f}%")
print(f"  Avg retention at 12 months : {avg_12m:.1f}%")
if not np.isnan(avg_24m):
    print(f"  Avg retention at 24 months : {avg_24m:.1f}%")

best_cohort  = ret_df["12M"].idxmax()
worst_cohort = ret_df["12M"].idxmin()
print(f"\n  Best 12M cohort  : {best_cohort} ({ret_df.loc[best_cohort,'12M']:.1f}%)")
print(f"  Worst 12M cohort : {worst_cohort} ({ret_df.loc[worst_cohort,'12M']:.1f}%)")
print("=" * 55)
