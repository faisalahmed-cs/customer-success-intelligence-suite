"""
churn_model.py
--------------
Trains a logistic-regression churn predictor on the customer dataset.
Outputs:
  • charts/churn_feature_importance.png  — top drivers of churn
  • charts/churn_risk_distribution.png   — probability distribution
  • data/at_risk_accounts.csv            — ranked watchlist for CSMs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings, os
warnings.filterwarnings("ignore")

os.makedirs("charts", exist_ok=True)
os.makedirs("data",   exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/customers.csv")

# ── Feature engineering ───────────────────────────────────────────────────────
# Days since last engagement
df["last_engagement"] = pd.to_datetime(df["last_engagement"])
df["days_since_engagement"] = (
    pd.Timestamp("2025-04-01") - df["last_engagement"]
).dt.days

# Encode QBR as integer
df["qbr_int"] = df["qbr_completed"].astype(int)

# Months since contract start (tenure)
df["contract_start"] = pd.to_datetime(df["contract_start"])
df["tenure_months"] = (
    (pd.Timestamp("2025-04-01") - df["contract_start"]).dt.days / 30
).round(1)

FEATURES = [
    "login_freq_monthly",
    "support_tickets_monthly",
    "nps_score",
    "qbr_int",
    "seat_utilisation",
    "days_since_engagement",
    "tenure_months",
    "mrr",
]

FEATURE_LABELS = {
    "login_freq_monthly":       "Login Frequency (monthly)",
    "support_tickets_monthly":  "Support Ticket Volume",
    "nps_score":                "NPS Score",
    "qbr_int":                  "QBR Completed",
    "seat_utilisation":         "Seat Utilisation Rate",
    "days_since_engagement":    "Days Since Last Engagement",
    "tenure_months":            "Tenure (months)",
    "mrr":                      "Monthly Recurring Revenue",
}

X = df[FEATURES]
y = df["churned"].astype(int)

# ── Train / test split & scaling ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Model ─────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print("=" * 55)
print("  CHURN PREDICTION MODEL — PERFORMANCE SUMMARY")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
print(f"  ROC-AUC Score : {auc:.3f}")
print("=" * 55)

# ── Feature importance chart ──────────────────────────────────────────────────
coefs      = model.coef_[0]
feat_imp   = pd.DataFrame({
    "feature": FEATURES,
    "label":   [FEATURE_LABELS[f] for f in FEATURES],
    "coef":    coefs,
}).sort_values("coef")

PALETTE = {"churn": "#E05252", "protect": "#4ABFA0"}
colors = [PALETTE["churn"] if c > 0 else PALETTE["protect"] for c in feat_imp["coef"]]

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

bars = ax.barh(feat_imp["label"], feat_imp["coef"], color=colors,
               edgecolor="none", height=0.6)

# Value labels
for bar, val in zip(bars, feat_imp["coef"]):
    x_pos = val + (0.01 if val >= 0 else -0.01)
    ha    = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", ha=ha,
            color="white", fontsize=9, fontweight="bold")

ax.axvline(0, color="#555", linewidth=1.2, linestyle="--")
ax.set_xlabel("Model Coefficient  (positive = increases churn risk)",
              color="#AAAAAA", fontsize=10)
ax.set_title("Churn Drivers — Feature Importance",
             color="white", fontsize=14, fontweight="bold", pad=16)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_visible(False)

legend_patches = [
    mpatches.Patch(color=PALETTE["churn"],   label="Increases churn risk"),
    mpatches.Patch(color=PALETTE["protect"], label="Reduces churn risk"),
]
ax.legend(handles=legend_patches, loc="lower right",
          facecolor="#1C1F26", edgecolor="#333", labelcolor="white", fontsize=9)

plt.tight_layout()
plt.savefig("charts/churn_feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✅  Saved → charts/churn_feature_importance.png")

# ── Churn probability distribution ───────────────────────────────────────────
all_proba = model.predict_proba(scaler.transform(X))[:, 1]
df["churn_probability"] = all_proba

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#0F1117")

retained = df[df["churned"] == False]["churn_probability"]
churned  = df[df["churned"] == True ]["churn_probability"]

ax.hist(retained, bins=30, alpha=0.7, color="#4ABFA0", label="Retained",  edgecolor="none")
ax.hist(churned,  bins=30, alpha=0.7, color="#E05252", label="Churned",   edgecolor="none")

ax.axvline(0.5, color="#FFD166", linestyle="--", linewidth=1.5, label="Decision threshold (0.5)")

ax.set_xlabel("Predicted Churn Probability", color="#AAAAAA", fontsize=11)
ax.set_ylabel("Number of Accounts",          color="#AAAAAA", fontsize=11)
ax.set_title("Churn Probability Distribution — Retained vs Churned",
             color="white", fontsize=14, fontweight="bold", pad=16)
ax.tick_params(colors="white")
ax.legend(facecolor="#1C1F26", edgecolor="#333", labelcolor="white", fontsize=10)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("charts/churn_risk_distribution.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("  ✅  Saved → charts/churn_risk_distribution.png")

# ── At-risk account watchlist ─────────────────────────────────────────────────
watchlist = (
    df[df["churned"] == False]          # active accounts only
    .copy()
    .assign(risk_tier=pd.cut(
        df[df["churned"] == False]["churn_probability"],
        bins=[0, 0.35, 0.60, 1.01],
        labels=["Green — Healthy", "Yellow — Monitor", "Red — Intervene Now"]
    ))
    .sort_values("churn_probability", ascending=False)
    [[
        "customer_id", "company_name", "industry", "tier", "mrr",
        "churn_probability", "risk_tier",
        "login_freq_monthly", "support_tickets_monthly",
        "nps_score", "qbr_completed", "seat_utilisation",
        "days_since_engagement"
    ]]
)

watchlist.to_csv("data/at_risk_accounts.csv", index=False)
red    = (watchlist["risk_tier"] == "Red — Intervene Now").sum()
yellow = (watchlist["risk_tier"] == "Yellow — Monitor").sum()
green  = (watchlist["risk_tier"] == "Green — Healthy").sum()

print(f"\n  AT-RISK WATCHLIST BREAKDOWN (active accounts only)")
print(f"  🔴  Red    (Intervene Now) : {red}")
print(f"  🟡  Yellow (Monitor)       : {yellow}")
print(f"  🟢  Green  (Healthy)       : {green}")
print(f"\n  Saved → data/at_risk_accounts.csv")
print(f"\n  Top 10 highest-risk active accounts:")
print(watchlist.head(10)[
    ["company_name", "mrr", "churn_probability", "risk_tier",
     "login_freq_monthly", "nps_score"]
].to_string(index=False))
