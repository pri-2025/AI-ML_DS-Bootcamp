# AI, Machine Learning & Data Science — End-to-End Projects

A collection of three end-to-end machine learning projects built as part of the **Complete A.I. & Machine Learning, Data Science Bootcamp** by Zero to Mastery.

Each project follows the full ML workflow — problem definition, EDA, preprocessing, modelling, evaluation, and prediction — on real-world datasets.

---

## Projects

| # | Project | Type | Tools | Key Metric |
|---|---------|------|-------|------------|
| 01 | [Heart Disease Classification](#-01---heart-disease-classification) | Binary Classification | scikit-learn, pandas, matplotlib | ~88% Accuracy |
| 02 | [Bulldozer Price Regression](#-02---bulldozer-price-regression) | Regression | scikit-learn, RandomForest, pandas | ~0.24 RMSLE |
| 03 | [Dog Breed Identification](#-03---dog-breed-identification) | Multi-class Image Classification | TensorFlow, TF Hub, MobileNetV2 | Transfer Learning |

---

## 01 — Heart Disease Classification

**Goal:** Predict whether a patient has heart disease based on clinical features.

**Dataset:** [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-ms) — 303 patients, 13 features.

**Approach:**
- Exploratory data analysis: class distribution, feature correlations, crosstabs
- Compared 3 models: Logistic Regression, KNN, Random Forest
- Hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`
- Final evaluation: ROC-AUC, Confusion Matrix, Classification Report, Cross-validated F1

[`01-heart-disease-classification/`](./01-heart-disease-classification/)

---

## 02 — Bulldozer Price Regression

**Goal:** Predict the auction sale price of bulldozers based on equipment specs and sale history.

**Dataset:** [Kaggle Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) — 400,000+ records, 50+ features.

**Approach:**
- Temporal feature engineering (year, month, day extracted from `saledate`)
- Time-based train/validation split (year 2012 held out as validation)
- Categorical encoding using pandas `Categorical` + `.cat.codes`
- Missing value imputation with median + missing indicator columns
- `RandomForestRegressor` with `RandomizedSearchCV` tuning
- Evaluation: MAE, RMSLE, R²

[`02-bulldozer-price-regression/`](./02-bulldozer-price-regression/)

---

## 03 — Dog Breed Identification

**Goal:** Identify the breed of a dog from a photo. 120 possible breeds.

**Dataset:** [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) — 10,000+ labelled images.

**Approach:**
- Images preprocessed into normalized `(224, 224, 3)` float tensors
- One-hot encoded labels across 120 breeds
- Transfer learning via **MobileNetV2** (pretrained on ImageNet) from TensorFlow Hub
- Custom Dense(120, softmax) output layer trained on top
- Callbacks: TensorBoard + EarlyStopping
- Trained on full dataset (~10,000 images), predictions submitted to Kaggle

[`03-dog-breed-identification/`](./03-dog-breed-identification/)

---

## 🛠️ Tech Stack

- **Languages:** Python 3
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML:** scikit-learn
- **Deep Learning:** TensorFlow 2.x, TensorFlow Hub
- **Environment:** Jupyter Notebook / Google Colab

---

## Setup

```bash
git clone https://github.com/pri-2025/AI-ML-DS-Bootcamp.git
cd AI-ML-DS-Bootcamp
pip install -r requirements.txt
```

Open any notebook in Jupyter or Google Colab and run all cells.

> **Note:** The dog breed project requires a GPU runtime (use Google Colab with GPU enabled). The bulldozer dataset is large — download it from Kaggle and place it in `02-bulldozer-price-regression/data/`.

---

## Certificate

[Complete A.I. & Machine Learning, Data Science Bootcamp — Zero to Mastery](https://www.udemy.com/certificate/UC-0bb14cb5-5169-4369-aefc-873c94daf6df)

---

## Connect

- **LinkedIn:** [linkedin.com/in/prisha-b-parikh](https://linkedin.com/in/prisha-b-parikh/)
- **GitHub:** [github.com/pri-2025](https://github.com/pri-2025)
