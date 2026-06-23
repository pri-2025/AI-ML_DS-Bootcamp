# Dog Breed Identification

Identifying the breed of a dog from a photo using deep learning and transfer learning — a multi-class image classification problem from Kaggle.

---

## Problem

> Given an image of a dog, which of 120 possible breeds does it belong to?

**Type:** Multi-class Image Classification (120 classes)

**Competition:** [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

---

## Dataset

- **Source:** Kaggle Dog Breed Identification
- **Training set:** ~10,000 labelled images across 120 breeds
- **Test set:** ~10,000 unlabelled images (for Kaggle submission)
- **Labels:** provided as a CSV with image `id` and `breed` columns

---

## Evaluation Metric

**Multi-class log loss** — Kaggle requires a CSV of prediction probabilities for all 120 breeds per image (not just the predicted class).

---

## Workflow

### Data Preparation
- Constructed full image file paths from label CSV IDs
- One-hot encoded 120 breed labels as boolean arrays
- 80/20 train/validation split using `sklearn.model_selection.train_test_split`

### Image Preprocessing (`process_image`)
Each image is processed as follows:
1. Read file from disk with `tf.io.read_file`
2. Decode JPEG → 3-channel RGB Tensor
3. Normalize pixel values from `[0, 255]` → `[0.0, 1.0]`
4. Resize to `(224, 224)` — required input size for MobileNetV2

### Batching (`create_data_batches`)
- Training data: shuffled before batching (prevents learning order)
- Validation/test data: not shuffled (maintains deterministic evaluation)
- Batch size: 32

### Model Architecture
```
Input: (None, 224, 224, 3)
    ↓
MobileNetV2 (pretrained on ImageNet, weights frozen)  ← Feature Extractor
    ↓
Dense(120, activation='softmax')                       ← Classification Head
```

- **Loss:** CategoricalCrossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

### Training
- **Phase 1:** Trained on 1000-image subset to verify pipeline
- **Phase 2:** Trained on full ~10,000 image dataset
- **Callbacks:**
  - `TensorBoard` — logs training/validation metrics per epoch
  - `EarlyStopping` (patience=3) — stops when accuracy plateaus

### Prediction & Submission
- `predict()` returns a `(n, 120)` probability array
- `np.argmax()` per row → predicted breed index → breed name
- Submitted as a CSV with columns: `id, breed_1, breed_2, ..., breed_120`

---

## Results

| Training Set Size | Val Accuracy (approx.) |
|------------------|------------------------|
| 1,000 images | ~70% (overfits) |
| Full (~10,000 images) | ~90%+ |

> Overfitting on 1000 images is expected — with 120 classes and ~800 training images, the model sees fewer than 7 images per breed on average. Full dataset training resolves this.

---

## Files

```
03-dog-breed-identification/
├── dog-breed-identification.ipynb       # Main notebook (fully run)
├── predictions_submission.csv          # Kaggle submission file
└── README.md
```

> The full image dataset (~750MB) and trained model weights are not included due to file size. Download the dataset from [Kaggle](https://www.kaggle.com/c/dog-breed-identification/data) and use Google Drive for model storage (see notebook).

---

## Run

This notebook requires a **GPU runtime**.

1. Open in [Google Colab](https://colab.research.google.com/)
2. Set Runtime → Change runtime type → **GPU**
3. Mount Google Drive and upload the Kaggle dataset
4. Run all cells

```python
# Install TensorFlow Hub if not available
pip install tensorflow-hub
```