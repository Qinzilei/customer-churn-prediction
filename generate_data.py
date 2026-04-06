"""
Synthetic e-commerce behavioral dataset generator for customer churn prediction.
Produces realistic feature distributions with meaningful churn signals.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# --- Customer segments ---
# 0: at-risk (30%), 1: loyal (50%), 2: new (20%)
segment = np.random.choice([0, 1, 2], size=N, p=[0.30, 0.50, 0.20])

def seg(v0, v1, v2):
    return np.where(segment == 0, v0, np.where(segment == 1, v1, v2))

# Features
days_since_last_purchase = np.clip(
    seg(
        np.random.exponential(120, N),
        np.random.exponential(20, N),
        np.random.exponential(60, N),
    ),
    1, 365,
).astype(int)

purchase_frequency = np.clip(
    seg(
        np.random.poisson(1, N),
        np.random.poisson(8, N),
        np.random.poisson(2, N),
    ),
    0, 50,
)

avg_session_duration_min = np.clip(
    seg(
        np.random.normal(3, 2, N),
        np.random.normal(18, 5, N),
        np.random.normal(8, 3, N),
    ),
    0.5, 60,
).round(1)

cart_abandonment_rate = np.clip(
    seg(
        np.random.beta(5, 2, N),
        np.random.beta(1, 5, N),
        np.random.beta(3, 3, N),
    ),
    0, 1,
).round(3)

total_spend_usd = np.clip(
    seg(
        np.random.exponential(50, N),
        np.random.exponential(500, N),
        np.random.exponential(80, N),
    ),
    0, 5000,
).round(2)

num_support_tickets = np.clip(
    seg(
        np.random.poisson(2.5, N),
        np.random.poisson(0.3, N),
        np.random.poisson(0.8, N),
    ),
    0, 15,
)

pages_per_session = np.clip(
    seg(
        np.random.normal(3, 1.5, N),
        np.random.normal(12, 3, N),
        np.random.normal(6, 2, N),
    ),
    1, 40,
).round(1)

discount_usage_rate = np.clip(
    seg(
        np.random.beta(4, 3, N),
        np.random.beta(2, 5, N),
        np.random.beta(3, 4, N),
    ),
    0, 1,
).round(3)

email_open_rate = np.clip(
    seg(
        np.random.beta(1, 5, N),
        np.random.beta(4, 2, N),
        np.random.beta(2, 3, N),
    ),
    0, 1,
).round(3)

preferred_device = np.where(
    segment == 0,
    np.random.choice(["mobile", "desktop", "tablet"], N, p=[0.6, 0.3, 0.1]),
    np.where(
        segment == 1,
        np.random.choice(["mobile", "desktop", "tablet"], N, p=[0.4, 0.5, 0.1]),
        np.random.choice(["mobile", "desktop", "tablet"], N, p=[0.5, 0.35, 0.15]),
    ),
)

membership_tier = np.where(
    segment == 0,
    np.random.choice(["bronze", "silver", "gold", "platinum"], N, p=[0.6, 0.25, 0.12, 0.03]),
    np.where(
        segment == 1,
        np.random.choice(["bronze", "silver", "gold", "platinum"], N, p=[0.15, 0.30, 0.35, 0.20]),
        np.random.choice(["bronze", "silver", "gold", "platinum"], N, p=[0.55, 0.30, 0.12, 0.03]),
    ),
)

# --- Churn label (probabilistic, based on features) ---
churn_prob = (
    0.30
    + 0.25 * (days_since_last_purchase / 365)
    + 0.15 * (1 - np.clip(purchase_frequency / 10, 0, 1))
    + 0.10 * cart_abandonment_rate
    - 0.15 * (email_open_rate)
    - 0.10 * np.clip(total_spend_usd / 1000, 0, 1)
    + 0.05 * (num_support_tickets / 15)
    - 0.05 * (pages_per_session / 40)
)
churn_prob = np.clip(churn_prob, 0.03, 0.97)
churned = (np.random.rand(N) < churn_prob).astype(int)

# --- Inject ~5% missing values ---
df = pd.DataFrame({
    "days_since_last_purchase": days_since_last_purchase,
    "purchase_frequency":       purchase_frequency,
    "avg_session_duration_min": avg_session_duration_min,
    "cart_abandonment_rate":    cart_abandonment_rate,
    "total_spend_usd":          total_spend_usd,
    "num_support_tickets":      num_support_tickets,
    "pages_per_session":        pages_per_session,
    "discount_usage_rate":      discount_usage_rate,
    "email_open_rate":          email_open_rate,
    "preferred_device":         preferred_device,
    "membership_tier":          membership_tier,
    "churned":                  churned,
})

rng = np.random.default_rng(99)
for col in ["avg_session_duration_min", "cart_abandonment_rate", "email_open_rate",
            "total_spend_usd", "pages_per_session"]:
    mask = rng.random(N) < 0.05
    df.loc[mask, col] = np.nan

df.to_csv("data/ecommerce_churn.csv", index=False)
print(f"Dataset saved: {len(df)} rows, churn rate = {df.churned.mean():.2%}")
