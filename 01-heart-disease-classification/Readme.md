# Heart Disease Classification

Predicting the presence of heart disease in a patient using clinical features — a binary classification problem.

---

## Problem

> Given a set of clinical parameters about a patient, can we predict whether or not they have heart disease?

**Type:** Binary Classification (0 = No heart disease, 1 = Heart disease)

---

## Dataset

- **Source:** [UCI Heart Disease Dataset via Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-ms)
- **Size:** 303 patients, 14 columns (13 features + 1 target)

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type |
| `target` | 1 = heart disease, 0 = no heart disease |

---

## Evaluation Metric

**Target:** ≥ 85% accuracy on the test set.

Additional metrics: ROC-AUC, Precision, Recall, F1-Score (cross-validated).

---

## Workflow

1. **EDA** — class distribution, sex vs. target crosstab, age vs. max heart rate scatter, correlation heatmap
2. **Modelling** — compared Logistic Regression, KNN, Random Forest
3. **Hyperparameter Tuning** — `RandomizedSearchCV` → `GridSearchCV`
4. **Evaluation** — ROC curve, confusion matrix, classification report, 5-fold cross-validation

---

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression (baseline) | ~82% |
| KNN | ~68% |
| Random Forest | ~83% |
| **Logistic Regression (tuned)** | **~88%** |

Cross-validated metrics (5-fold) on tuned Logistic Regression:

| Metric | Score |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~85% |
| Recall | ~90% |
| F1 | ~87% |

---

## Files

```
01-heart-disease-classification/
├── heart-disease-classification.ipynb   # Main notebook (fully run)
├── heart-disease.csv                    # Dataset
└── README.md
```

---

## Run

```bash
jupyter notebook heart-disease-classification.ipynb
```

No special dependencies beyond the standard data science stack (numpy, pandas, matplotlib, seaborn, scikit-learn).