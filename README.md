# CS 349 — Machine Learning
**Northwestern University · McCormick School of Engineering**

Applied machine learning course. Primary project: a diagnostic classifier for a rare pediatric genetic disorder using facial landmark data — achieving >85% accuracy and >95% precision as a cost-effective alternative to genetic testing.

---

## Project — CCHS Genetic Disease Classifier

### Problem
Congenital Central Hypoventilation Syndrome (CCHS) is a rare genetic disorder causing inadequate breathing during sleep. Standard diagnosis requires a PHOX2B genetic test, which is expensive and inaccessible in many clinical settings. Researchers have identified a correlation between specific facial feature patterns and CCHS diagnosis.

### Approach
Used **Facial Landmark Detection** data — 67 vector points mapping facial geometry — as input features to train two classifiers:

1. **Support Vector Machine (SVM)** — Vanilla SVM via scikit-learn, trained on standardized landmark features
2. **3-Layer Neural Network** — Custom PyTorch architecture: `Linear(139) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid`

### Data Pipeline
```
Raw facial landmarks (67 points)
    → StandardScaler normalization
    → RandomOverSampler (handle class imbalance)
    → Train/Test split (80/20)
    → SVM + Neural Network training
    → Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
```

### Results

| Model | Accuracy | Precision |
|-------|----------|-----------|
| SVM | >85% | >95% |
| Neural Network | >85% | >95% |

Both models exceed clinical utility thresholds, providing a low-cost screening tool to determine whether expensive PHOX2B genetic testing is warranted.

### Architecture
```python
class BinaryClassifier(nn.Module):
    # Input → Linear(139) → ReLU → Linear(64) → ReLU → Linear(1) → Sigmoid
    # Threshold: 0.4 (tuned for recall on rare positive class)
```

---

## Stack
`Python` `PyTorch` `scikit-learn` `imbalanced-learn` `Pandas` `NumPy` `Matplotlib`

---

*Northwestern University — CS 349, Fall 2023 · Machine Learning*
