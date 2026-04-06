"""
Standalone EDA visualisation script.
Generates all plots to plots/ for README embedding.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

Path("plots").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.15)
plt.rcParams["figure.dpi"] = 130

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
PALETTE = [BLUE, ORANGE]

df = pd.read_csv("data/ecommerce_churn.csv")
NUM_COLS = [
    "days_since_last_purchase", "purchase_frequency", "avg_session_duration_min",
    "cart_abandonment_rate", "total_spend_usd", "num_support_tickets",
    "pages_per_session", "email_open_rate", "discount_usage_rate",
]

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Churn distribution (donut)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
counts = df["churned"].value_counts().sort_index()
labels = [f"Retained\n{counts[0]:,} ({counts[0]/len(df)*100:.1f}%)",
          f"Churned\n{counts[1]:,} ({counts[1]/len(df)*100:.1f}%)"]
wedges, texts = ax.pie(
    counts, labels=labels, colors=PALETTE,
    startangle=90, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    textprops=dict(fontsize=13),
)
ax.set_title("Target Class Distribution", fontsize=15, pad=16)
centre_circle = plt.Circle((0, 0), 0.45, color="white")
ax.add_patch(centre_circle)
ax.text(0, 0, f"n = {len(df):,}", ha="center", va="center", fontsize=12, color="grey")
plt.tight_layout()
plt.savefig("plots/churn_distribution.png", bbox_inches="tight")
plt.close()
print("✓ churn_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Box plots: numeric features by churn
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for ax, col in zip(axes, NUM_COLS):
    data_plot = df[[col, "churned"]].dropna()
    data_plot["churn_label"] = data_plot["churned"].map({0: "Retained", 1: "Churned"})
    sns.boxplot(
        x="churn_label", y=col, data=data_plot,
        palette={"Retained": BLUE, "Churned": ORANGE},
        linewidth=1.2, flierprops=dict(marker=".", alpha=0.3, markersize=3),
        ax=ax,
    )
    ax.set_title(col.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.suptitle("Numeric Feature Distributions by Churn Status (Box Plots)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plots/boxplots_by_churn.png", bbox_inches="tight")
plt.close()
print("✓ boxplots_by_churn.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Violin plots: top 4 most discriminative features
# ─────────────────────────────────────────────────────────────────────────────
top4 = ["days_since_last_purchase", "purchase_frequency", "email_open_rate", "cart_abandonment_rate"]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, col in zip(axes, top4):
    data_plot = df[[col, "churned"]].dropna()
    data_plot["churn_label"] = data_plot["churned"].map({0: "Retained", 1: "Churned"})
    sns.violinplot(
        x="churn_label", y=col, data=data_plot,
        palette={"Retained": BLUE, "Churned": ORANGE},
        inner="quartile", linewidth=1.2, ax=ax,
    )
    ax.set_title(col.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel(col.replace("_", " "))

plt.suptitle("Violin Plots — Top 4 Discriminative Features", fontsize=14, y=1.03)
plt.tight_layout()
plt.savefig("plots/violin_top_features.png", bbox_inches="tight")
plt.close()
print("✓ violin_top_features.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Scatter: recency vs frequency, coloured by churn
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for churn_val, label, color, marker in [(0, "Retained", BLUE, "o"), (1, "Churned", ORANGE, "x")]:
    sub = df[df["churned"] == churn_val]
    ax.scatter(
        sub["days_since_last_purchase"], sub["purchase_frequency"],
        c=color, label=label, alpha=0.35, s=18, marker=marker, linewidths=0.8,
    )
ax.set_xlabel("Days Since Last Purchase")
ax.set_ylabel("Purchase Frequency")
ax.set_title("Recency vs Purchase Frequency — Coloured by Churn", fontsize=13)
ax.legend(markerscale=2, fontsize=11)
plt.tight_layout()
plt.savefig("plots/scatter_recency_frequency.png", bbox_inches="tight")
plt.close()
print("✓ scatter_recency_frequency.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Stacked bar: membership tier churn breakdown
# ─────────────────────────────────────────────────────────────────────────────
tier_order  = ["bronze", "silver", "gold", "platinum"]
tier_counts = df.groupby(["membership_tier", "churned"]).size().unstack(fill_value=0)
tier_counts = tier_counts.reindex(tier_order)
tier_pct    = tier_counts.div(tier_counts.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

tier_pct.plot(kind="bar", stacked=True, color=PALETTE, ax=axes[0],
              edgecolor="white", linewidth=0.8, rot=0)
axes[0].set_title("Churn vs Retained (%) by Membership Tier", fontsize=12)
axes[0].set_xlabel("Membership Tier")
axes[0].set_ylabel("Percentage (%)")
axes[0].legend(["Retained", "Churned"], loc="upper right")
for p in axes[0].patches:
    h = p.get_height()
    if h > 4:
        axes[0].text(
            p.get_x() + p.get_width() / 2,
            p.get_y() + h / 2,
            f"{h:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )

churn_rate_tier = df.groupby("membership_tier")["churned"].mean().reindex(tier_order) * 100
axes[1].bar(tier_order, churn_rate_tier, color=sns.color_palette("flare", 4), edgecolor="white")
axes[1].set_title("Churn Rate (%) by Membership Tier", fontsize=12)
axes[1].set_xlabel("Membership Tier")
axes[1].set_ylabel("Churn Rate (%)")
for i, (tier, val) in enumerate(zip(tier_order, churn_rate_tier)):
    axes[1].text(i, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("plots/membership_tier_churn.png", bbox_inches="tight")
plt.close()
print("✓ membership_tier_churn.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Heatmap: mean feature values, Churned vs Retained
# ─────────────────────────────────────────────────────────────────────────────
profile = df.groupby("churned")[NUM_COLS].mean()
profile.index = ["Retained", "Churned"]
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(13, 3.5))
sns.heatmap(
    profile_norm, annot=profile.round(2), fmt=".2f",
    cmap="RdYlGn_r", ax=ax, linewidths=0.5,
    annot_kws={"size": 10}, cbar_kws={"label": "Normalised mean"},
    xticklabels=[c.replace("_", "\n") for c in NUM_COLS],
)
ax.set_title("Customer Profile Heatmap — Mean Feature Values (normalised)", fontsize=13, pad=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig("plots/customer_profile_heatmap.png", bbox_inches="tight")
plt.close()
print("✓ customer_profile_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — KDE: spend & session duration by churn
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for churn_val, label, color in [(0, "Retained", BLUE), (1, "Churned", ORANGE)]:
    sub = df[df["churned"] == churn_val]
    sns.kdeplot(sub["total_spend_usd"].dropna(), ax=axes[0],
                label=label, color=color, fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(sub["avg_session_duration_min"].dropna(), ax=axes[1],
                label=label, color=color, fill=True, alpha=0.35, linewidth=2)

axes[0].set_title("Total Spend — KDE by Churn Status", fontsize=12)
axes[0].set_xlabel("Total Spend (USD)")
axes[0].set_ylabel("Density")
axes[0].legend()
axes[0].set_xlim(left=0)

axes[1].set_title("Avg Session Duration — KDE by Churn Status", fontsize=12)
axes[1].set_xlabel("Session Duration (minutes)")
axes[1].set_ylabel("Density")
axes[1].legend()
axes[1].set_xlim(left=0)

plt.tight_layout()
plt.savefig("plots/kde_spend_session.png", bbox_inches="tight")
plt.close()
print("✓ kde_spend_session.png")

print("\nAll EDA plots generated.")
