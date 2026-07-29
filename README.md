# Pima Indians Diabetes — Classification

Predicts whether a patient has diabetes using clinical measurements from the
[Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

## Key Skills Demonstrated
- **Models:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN
- **Class imbalance handling:** `class_weight="balanced"` for most models,
  plus a **SMOTE implementation written from scratch** (no extra library
  needed) applied before training KNN
- **Threshold tuning:** default 0.5 cutoff vs. the threshold that maximizes F1
- **Evaluation:** ROC curves, Precision-Recall curves, ROC-AUC, PR-AUC

## Project Structure
```
diabetes-ml/
├── data/
│   └── diabetes.csv        # raw dataset
├── outputs/                 # generated after running app.py
│   ├── model_results.csv
│   ├── roc_pr_curves.png
│   └── confusion_matrices.png
├── app.py                   # full pipeline: load -> clean -> train -> evaluate
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python app.py
```

## Approach

1. **Data cleaning** — columns like `Glucose`, `BMI`, `BloodPressure` etc.
   contain `0` values that are biologically impossible; these are treated as
   missing and replaced with the column median.
2. **Train/test split** — 80/20, stratified so both sets keep the same class
   ratio (~65% non-diabetic / 35% diabetic).
3. **Scaling** — `StandardScaler` is fit on the training set only.
4. **Imbalance handling** — two complementary techniques:
   - `class_weight="balanced"` for Logistic Regression, Decision Tree,
     Random Forest, and SVM (they support it natively).
   - A simple **SMOTE** function (nearest-neighbour interpolation) applied to
     the training data before fitting KNN, which has no `class_weight`
     option.
5. **Threshold tuning** — instead of the default 0.5 probability cutoff, the
   script scans thresholds from 0.05–0.95 and picks the one that maximizes
   F1-score, since in a medical context missing a diabetic patient
   (false negative) is usually worse than a false alarm.
6. **Evaluation** — ROC-AUC, PR-AUC, and F1 (both default and tuned
   threshold) are reported for every model, alongside ROC and
   Precision-Recall curve plots.

## Sample Results

| Model               | ROC-AUC | PR-AUC | F1 (0.5) | F1 (best) | Best Threshold |
|----------------------|---------|--------|----------|-----------|-----------------|
| Random Forest         | 0.815   | 0.669  | 0.614    | 0.691     | 0.25            |
| SVM                    | 0.814   | 0.678  | 0.614    | 0.688     | 0.31            |
| Logistic Regression    | 0.813   | 0.673  | 0.650    | 0.696     | 0.35            |
| KNN                    | 0.793   | 0.634  | 0.677    | 0.677     | 0.45            |
| Decision Tree          | 0.771   | 0.573  | 0.611    | 0.657     | 0.40            |

(Numbers will vary slightly depending on random seed / package versions.)

**Takeaway:** Random Forest, SVM, and Logistic Regression perform closely at
the top (~0.81 ROC-AUC). Tuning the classification threshold instead of
using the default 0.5 noticeably improves F1 for every model — showing why
threshold tuning matters more than model choice alone when classes are
imbalanced.
