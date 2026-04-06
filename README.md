# Customer Churn Prediction — E-Commerce Platform

End-to-end machine learning project that predicts customer churn for an e-commerce business using behavioural data. Includes synthetic data generation, full preprocessing pipeline, three classifiers, 5-fold cross-validation, and visual feature importance analysis.

---

## Results Summary

| Model | CV Accuracy | CV F1 | Test Accuracy | Test F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.6787 ± 0.0040 | 0.5298 ± 0.0066 | **0.6660** | **0.4893** | **0.6750** |
| Random Forest | 0.6633 ± 0.0084 | 0.5017 ± 0.0113 | 0.6550 | 0.4651 | 0.6482 |
| K-Nearest Neighbors | 0.6493 ± 0.0164 | 0.4857 ± 0.0239 | 0.6420 | 0.4641 | 0.6342 |

> **Best model:** Logistic Regression — Test Accuracy **0.6660**, Test F1 **0.4893**, AUC-ROC **0.6750**
> All numbers are produced by executing the notebook end-to-end; no post-hoc editing.

---

## Dataset

A synthetic e-commerce behavioural dataset (`data/ecommerce_churn.csv`) with **5,000 customers** and **12 features**. Generated via `generate_data.py` with three realistic customer segments (at-risk, loyal, new). Churn rate ≈ 37%.

| Feature | Description |
|---|---|
| `days_since_last_purchase` | Days elapsed since the most recent order |
| `purchase_frequency` | Number of purchases in the observation window |
| `avg_session_duration_min` | Mean browsing session length (minutes) |
| `cart_abandonment_rate` | Fraction of carts left without checkout |
| `total_spend_usd` | Cumulative spend in USD |
| `num_support_tickets` | Support/complaint tickets opened |
| `pages_per_session` | Average pages viewed per session |
| `discount_usage_rate` | Proportion of orders using a discount code |
| `email_open_rate` | Marketing email open rate |
| `preferred_device` | Device type (mobile / desktop / tablet) |
| `membership_tier` | Loyalty tier (bronze / silver / gold / platinum) |
| `churned` | Target: 1 = churned, 0 = retained |

~5% of values in continuous features are randomly set to `NaN` to simulate real-world data quality issues.

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── ecommerce_churn.csv         # Synthetic dataset (5 000 rows)
├── plots/
│   ├── eda_distributions.png
│   ├── eda_categorical.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── cv_comparison.png
│   ├── feature_importance.png
│   └── permutation_importance.png
├── customer_churn_prediction.ipynb  # Main notebook (executed, with outputs)
├── generate_data.py                 # Dataset generator
├── build_notebook.py                # Notebook builder / runner script
├── requirements.txt
└── README.md
```

---

## Methodology

### Preprocessing
- **Missing values** — `SimpleImputer(strategy="median")` applied _inside_ each sklearn `Pipeline` to prevent data leakage during cross-validation.
- **Categorical encoding** — membership tier → ordinal integer (bronze=0 … platinum=3); preferred device → one-hot encoded with `pd.get_dummies`.
- **Standardisation** — `StandardScaler` applied to Logistic Regression and KNN pipelines; Random Forest is scale-invariant.

### Models
| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `C=1.0`, `max_iter=1000`, `solver='lbfgs'` |
| Random Forest | `n_estimators=200`, `max_depth=None`, `n_jobs=-1` |
| K-Nearest Neighbors | `n_neighbors=11`, `metric='euclidean'` |

### Evaluation
- **Hold-out test set** — 20% stratified split (1 000 samples)
- **Cross-validation** — `StratifiedKFold(n_splits=5, shuffle=True)` on training set
- **Metrics** — Accuracy, F1-Score (churn class), AUC-ROC
- **Feature importance** — Gini importance (RF), absolute coefficients (LR), and model-agnostic permutation importance

---

## Key Insights

1. **`days_since_last_purchase`** is the single strongest churn predictor — recency matters most.
2. **`purchase_frequency`** is the second most important signal; infrequent buyers churn at a much higher rate.
3. **`email_open_rate`** acts as a *protective* factor: engaged subscribers are far less likely to churn.
4. **`cart_abandonment_rate`** and **`num_support_tickets`** serve as early-warning signals.
5. Logistic Regression slightly outperforms tree-based and distance-based approaches on this dataset, suggesting largely linear separability with a moderate class overlap.

### Business Recommendations
- Launch win-back campaigns for customers with `days_since_last_purchase > 90`.
- Automate email re-engagement for customers with `email_open_rate < 0.15`.
- Prioritise loyalty programme upgrades to move bronze/silver customers to higher tiers.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-generate the dataset
python generate_data.py

# 3. Open the notebook
jupyter notebook customer_churn_prediction.ipynb
```

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
notebook
nbformat
nbconvert
```
