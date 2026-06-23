# Bulldozer Price Regression

Predicting the auction sale price of bulldozers using historical sales data — a time series regression problem from Kaggle.

---

## Problem

> How well can we predict the future sale price of a bulldozer, given its characteristics and previous auction data?

**Type:** Regression (predicting a continuous value — sale price in USD)

**Competition:** [Kaggle Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers)

---

## Dataset

- **Source:** Kaggle Bluebook for Bulldozers
- **Size:** 400,000+ records across training + validation sets, 50+ features
- **Files used:**
  - `TrainAndValid.csv` — labelled training data (sales up to end of 2011)
  - `Test.csv` — unlabelled test data (sales from 2012)

Key features include: `YearMade`, `MachineHoursCurrentMeter`, `UsageBand`, `Enclosure`, `ProductSize`, `fiModelDesc`, `state`, and `saledate`.

---

## Evaluation Metric

**RMSLE** — Root Mean Squared Log Error (Kaggle's official metric for this competition).

> RMSLE penalises underestimates more than overestimates, which is appropriate here since underpredicting an expensive machine's price is a bigger error.

---

## Workflow

1. **Date Parsing** — extracted `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, `saleDayOfYear` from `saledate`
2. **Sorting** — sorted by `saledate` ascending (important for time series data integrity)
3. **Categorical Encoding** — converted string columns to pandas `Categorical`, then used `.cat.codes` for numerical codes
4. **Missing Value Imputation:**
   - Numerical → filled with median, added `_is_missing` boolean indicator column
   - Categorical → filled with -1 (unknown category code)
5. **Train/Val Split** — used 2012 sales as validation (time-based split, not random)
6. **Modelling** — `RandomForestRegressor` baseline → tuned with `RandomizedSearchCV`
7. **Feature Importance** — identified top drivers of price

---

## Results

| Model | Training R² | Validation RMSLE |
|-------|------------|-----------------|
| RandomForest (baseline, 1000 samples) | ~0.88 | ~0.36 |
| **RandomForest (tuned, full data)** | **~0.96** | **~0.24** |

**Top features by importance:** `YearMade`, `saleYear`, `ProductSize`, `fiProductClassDesc`, `Enclosure`

---

## Files

```
02-bulldozer-price-regression/
├── bulldozer-price-regression.ipynb     # Main notebook (fully run)
├── data/
│   └── (download from Kaggle — see below)
├── test_predictions.csv                 # Final predictions for Kaggle submission
└── README.md
```

---

## ▶️ Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/bluebook-for-bulldozers/data) and place files inside `data/bluebook-for-bulldozers/`
2. Run the notebook:

```bash
jupyter notebook bulldozer-price-regression.ipynb
```

> **Note:** Training on the full dataset (~400K rows) with `n_estimators=40` takes approximately 10–15 minutes on CPU.