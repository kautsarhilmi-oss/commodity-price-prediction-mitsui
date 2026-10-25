# 📈 Commodity Price Return Prediction — MITSUI&CO. Challenge

[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Domain](https://img.shields.io/badge/Domain-Financial%20Markets-blue)]()
[![Targets](https://img.shields.io/badge/Targets-424-orange)]()
[![Val MAE](https://img.shields.io/badge/Val%20MAE-0.0228-brightgreen)]()

## Overview

Multi-target time-series regression to predict future commodity returns
across 424 target pairs derived from global financial markets —
London Metal Exchange (LME), Japan Exchange Group (JPX),
US Stock markets, and Forex (FX).

Based on the **MITSUI&CO. Commodity Prediction Challenge** on Kaggle,
hosted by MITSUI & CO., LTD. — one of Japan's largest trading companies.

- **Task:** Predict 1–4 day ahead returns for 424 commodity/asset pairs
- **Features:** 557 columns — commodity prices, FX rates, stock indices
- **Targets:** 424 regression targets (single asset returns + pair spreads)
- **Val MAE:** 0.0228 (time-series walk-forward validation)

---

## Dataset Structure

| File | Shape | Description |
|------|-------|-------------|
| `train.csv` | 1,961 × 558 | Historical daily prices (date_id 0–1960) |
| `test.csv` | 134 × 559 | Future dates to predict |
| `train_labels.csv` | 1,961 × 425 | Target returns for training |
| `target_pairs.csv` | 424 × 3 | Target definitions (pair + lag) |

**Target structure:**
- 106 unique asset pairs × 4 lag values (1, 2, 3, 4 days) = 424 targets
- Each target is a **daily return** or **spread return** between two assets
- Values are small floats centered near zero (mean ~0.0005, std ~0.013)

---

## Approach

### 1. Target Decomposition
Each of the 424 targets maps to a specific asset pair and lag.
For each target, I extract the relevant columns from the pair definition
and build features specific to that pair.

### 2. Lag Feature Engineering
For each asset column, constructed:
- Raw lag values (1, 2, 3, 4, 5, 10, 20 days)
- Rolling returns (1-day, 5-day, 10-day pct_change)
- Rolling statistics (5-day MA, 20-day MA, 10-day std)
- Spread features (col1 - col2, col1 / col2 ratio)

### 3. Walk-Forward Validation
Used 80/20 time-based split — training on earlier dates,
validating on later dates. This respects temporal order
and prevents data leakage, which is critical in financial forecasting.

### 4. Per-Target Modeling
Trained one `HistGradientBoostingRegressor` per target —
424 models total. Each model is specialized for its
specific asset pair and lag combination.

---

## Results

| Metric | Value |
|--------|-------|
| Overall Avg Val MAE | 0.022752 |
| Targets trained | 424 / 424 |
| Training time | ~8 minutes |

MAE by training progress:
| Checkpoint | Avg Val MAE |
|-----------|-------------|
| 50/424    | 0.013975    |
| 100/424   | 0.014546    |
| 150/424   | 0.016223    |
| 200/424   | 0.017466    |
| 250/424   | 0.018803    |
| 300/424   | 0.019913    |
| 350/424   | 0.021066    |
| 400/424   | 0.022278    |
| **424/424** | **0.022752** |

---

## Key Learnings

- **Multi-target regression at scale** — training 424 independent models
  is surprisingly manageable with efficient per-target feature construction
- **Temporal data leakage is a critical risk** — using future data
  even accidentally destroys model validity in time-series
- **Spread features capture relative movement** — the difference and
  ratio between two assets often carries more signal than each alone
- **Financial returns are hard to predict** — MAE of ~0.02 on
  returns with std ~0.013 reflects the inherent noise in market data

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (HistGradientBoostingRegressor)
- Google Colab

---

## How to Run

```bash
pip install scikit-learn pandas numpy

# Download data from:
# https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge/data
# Place train.csv, test.csv, train_labels.csv, target_pairs.csv in same folder
```

Open `commodity_prediction_mitsui.ipynb` in Jupyter or Google Colab.

---

## Author

**Kautsar Hilmi** — Magister Teknik Informatika, Universitas Pamulang
*Data Science | Machine Learning | Financial Analytics*

[![GitHub](https://img.shields.io/badge/GitHub-kautsarhilmi--oss-black)](https://github.com/kautsarhilmi-oss)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kautsarhilmi-blue)](https://www.linkedin.com/in/kautsarhilmi/)
