"""
generate_data.py
----------------
Generates a realistic fictional SaaS customer dataset (500 rows) with
churn outcomes that correlate meaningfully to engagement signals.
Run this first before any analysis scripts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Reference data ───────────────────────────────────────────────────────────
N = 500

INDUSTRIES = [
    "E-Commerce / DTC", "SaaS / Tech", "Healthcare", "Financial Services",
    "Media & Entertainment", "Professional Services", "Logistics",
    "Education", "Real Estate", "Retail"
]

COMPANY_PREFIXES = [
    "Apex", "Nova", "Orbit", "Peak", "Crest", "Forge", "Lumis", "Vanta",
    "Drift", "Pulse", "Slate", "Nexus", "Prism", "Solus", "Helix", "Kova",
    "Zara", "Bloom", "Cedar", "Dune", "Echo", "Faro", "Glen", "Halo"
]
COMPANY_SUFFIXES = [
    "Inc", "Co", "Labs", "Group", "Studio", "Works", "HQ", "IO",
    "Digital", "Media", "Partners", "Solutions", "Ventures", "Systems"
]

TIERS = ["Starter", "Growth", "Scale", "Enterprise"]
TIER_MRR   = {"Starter": (500, 1500),   "Growth": (1500, 5000),
              "Scale":   (5000, 15000), "Enterprise": (15000, 50000)}
TIER_SEATS = {"Starter": (2, 10),       "Growth": (10, 30),
              "Scale":   (30, 80),      "Enterprise": (80, 300)}

# ── Helper – random date between two dates ────────────────────────────────────
def rand_date(start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

TODAY = datetime(2025, 4, 1)
WINDOW_START = datetime(2022, 1, 1)

# ── Build base features ───────────────────────────────────────────────────────
customer_ids      = [f"CUST-{str(i).zfill(4)}" for i in range(1, N + 1)]
company_names     = [f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
                     for _ in range(N)]
industries        = np.random.choice(INDUSTRIES, N)
tiers             = np.random.choice(TIERS, N, p=[0.30, 0.35, 0.25, 0.10])
contract_starts   = [rand_date(WINDOW_START, datetime(2024, 10, 1)) for _ in range(N)]
contract_values   = [round(random.uniform(*TIER_MRR[t]) * 12, 2) for t in tiers]
mrr               = [round(cv / 12, 2) for cv in contract_values]

# Seats licensed vs used
seats_licensed = [random.randint(*TIER_SEATS[t]) for t in tiers]
seats_used     = [max(1, int(sl * random.uniform(0.3, 1.05))) for sl in seats_licensed]
seats_used     = [min(su, sl) for su, sl in zip(seats_used, seats_licensed)]

# ── Engagement signals (these drive churn) ───────────────────────────────────
# Login frequency: healthy = 15-30/mo, at-risk = 1-8/mo
login_freq      = np.random.choice(
    [np.random.randint(1,  8),   # low
     np.random.randint(8,  15),  # medium
     np.random.randint(15, 31)], # high
    N, p=[0.25, 0.35, 0.40]
).tolist()
login_freq = [int(x) for x in login_freq]   # ensure plain int

# Support tickets: healthy = 0-3/mo, struggling = 8-20/mo
support_tickets = np.random.choice(
    [np.random.randint(0,  4),
     np.random.randint(4,  8),
     np.random.randint(8, 21)],
    N, p=[0.50, 0.30, 0.20]
).tolist()
support_tickets = [int(x) for x in support_tickets]

# NPS: -100 to 100 (simplified to 0-10 for readability)
nps_scores = np.clip(np.random.normal(7.0, 2.0, N), 0, 10).round(1).tolist()

# QBR completed
qbr_completed = np.random.choice([True, False], N, p=[0.65, 0.35]).tolist()

# Last engagement date
last_engagement = [
    rand_date(max(cs, TODAY - timedelta(days=180)), TODAY)
    for cs in contract_starts
]

# ── Churn logic (realistic ~20% rate) ────────────────────────────────────────
def churn_probability(login, tickets, nps, qbr, seats_u, seats_l):
    """
    Weighted logistic-style churn score.
    Low logins, high tickets, low NPS, no QBR, low seat utilisation → higher risk.
    """
    score = 0.0
    # Login frequency (strongest signal)
    if login < 5:    score += 0.40
    elif login < 10: score += 0.20
    else:            score -= 0.10

    # Support ticket volume
    if tickets > 12:  score += 0.30
    elif tickets > 6: score += 0.15
    else:             score -= 0.05

    # NPS
    if nps < 4:   score += 0.25
    elif nps < 6: score += 0.10
    else:         score -= 0.10

    # QBR
    if not qbr: score += 0.15

    # Seat utilisation
    util = seats_u / max(seats_l, 1)
    if util < 0.40: score += 0.10
    elif util > 0.80: score -= 0.05

    # Normalise to probability
    prob = 1 / (1 + np.exp(-score * 2))
    return float(np.clip(prob, 0.02, 0.95))

churn_probs = [
    churn_probability(login_freq[i], support_tickets[i], nps_scores[i],
                      qbr_completed[i], seats_used[i], seats_licensed[i])
    for i in range(N)
]

# Sample churn outcomes; scale so ≈20% churn overall
raw_churned = [np.random.random() < p for p in churn_probs]
churn_rate  = sum(raw_churned) / N
scale       = 0.20 / churn_rate if churn_rate > 0 else 1.0
churned     = []
for i, p in enumerate(churn_probs):
    adjusted = min(p * scale, 0.95)
    churned.append(bool(np.random.random() < adjusted))

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "customer_id":       customer_ids,
    "company_name":      company_names,
    "industry":          industries,
    "tier":              tiers,
    "contract_start":    [d.strftime("%Y-%m-%d") for d in contract_starts],
    "contract_value":    contract_values,
    "mrr":               mrr,
    "login_freq_monthly": login_freq,
    "support_tickets_monthly": support_tickets,
    "nps_score":         nps_scores,
    "qbr_completed":     qbr_completed,
    "seats_licensed":    seats_licensed,
    "seats_used":        seats_used,
    "seat_utilisation":  [(round(u/l, 2) if l > 0 else 0)
                          for u, l in zip(seats_used, seats_licensed)],
    "last_engagement":   [d.strftime("%Y-%m-%d") for d in last_engagement],
    "churned":           churned,
})

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("data/customers.csv", index=False)

actual_churn = df["churned"].mean()
print(f"✅  Dataset saved → data/customers.csv")
print(f"    Rows        : {len(df)}")
print(f"    Churn rate  : {actual_churn:.1%}")
print(f"    Industries  : {df['industry'].nunique()}")
print(f"    Tiers       : {df['tier'].value_counts().to_dict()}")
print("\nFirst 3 rows:")
print(df.head(3).to_string())
