"""
Pima Indians Diabetes - Classification Project
================================================
Goal: Predict whether a patient has diabetes (Outcome: 0/1) using clinical
measurements, and compare 5 classic ML algorithms while handling class
imbalance properly.

Models compared : Logistic Regression, Decision Tree, Random Forest, SVM, KNN
Imbalance fixes : class_weight='balanced'  AND  a simple custom SMOTE
Evaluation      : ROC curve, Precision-Recall curve, threshold tuning

Run:
    python app.py
Outputs (plots + metrics table) are saved inside the outputs/ folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix
)

DATA_PATH = "data/diabetes.csv"
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD + CLEAN DATA
# ---------------------------------------------------------------------------
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)

    # In this dataset, 0 is not a real medical value for these columns
    # (a person can't have 0 Glucose or 0 BMI) -> treat 0 as missing.
    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


# ---------------------------------------------------------------------------
# 2. SIMPLE SMOTE IMPLEMENTATION (no extra library needed)
# ---------------------------------------------------------------------------
def simple_smote(X, y, minority_class=1, k=5, random_state=RANDOM_STATE):
    """
    Oversamples the minority class by creating synthetic points between
    a real minority sample and one of its k nearest minority neighbours.
    This is the core idea behind SMOTE (Synthetic Minority Oversampling
    Technique), written from scratch so no extra package is required.
    """
    rng = np.random.RandomState(random_state)
    X_min = X[y == minority_class]
    X_maj_count = (y != minority_class).sum()
    n_to_create = X_maj_count - len(X_min)  # balance to 50/50

    if n_to_create <= 0:
        return X, y

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X_min))).fit(X_min)
    _, neighbors = nn.kneighbors(X_min)

    synthetic = []
    for _ in range(n_to_create):
        i = rng.randint(0, len(X_min))
        neighbor_idx = neighbors[i][rng.randint(1, neighbors.shape[1])]  # skip self (index 0)
        gap = rng.rand()
        new_point = X_min[i] + gap * (X_min[neighbor_idx] - X_min[i])
        synthetic.append(new_point)

    X_new = np.vstack([X, np.array(synthetic)])
    y_new = np.concatenate([y, np.full(n_to_create, minority_class)])
    return X_new, y_new


# ---------------------------------------------------------------------------
# 3. TRAIN ALL MODELS
# ---------------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=9),  # KNN has no class_weight -> SMOTE handles imbalance here
    }


def find_best_threshold(y_true, y_probs):
    """Pick the probability threshold that gives the best F1 score,
    instead of blindly using the default 0.5 cutoff."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def evaluate_model(name, y_true, y_probs):
    auc = roc_auc_score(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    best_thresh, best_f1 = find_best_threshold(y_true, y_probs)

    preds_default = (y_probs >= 0.5).astype(int)
    preds_tuned = (y_probs >= best_thresh).astype(int)

    return {
        "Model": name,
        "ROC-AUC": round(auc, 3),
        "PR-AUC": round(ap, 3),
        "F1 (thresh=0.5)": round(f1_score(y_true, preds_default), 3),
        "F1 (best thresh)": round(best_f1, 3),
        "Best Threshold": round(best_thresh, 2),
        "Precision (tuned)": round(precision_score(y_true, preds_tuned), 3),
        "Recall (tuned)": round(recall_score(y_true, preds_tuned), 3),
    }


# ---------------------------------------------------------------------------
# 4. PLOTS
# ---------------------------------------------------------------------------
def plot_roc_pr_curves(results_probs, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, probs in results_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probs):.2f})")

        prec, rec, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, probs):.2f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_pr_curves.png", dpi=150)
    plt.close()


def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    for ax, (name, model) in zip(axes, models.items()):
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df = load_data()

    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale features (important for Logistic Regression, SVM, KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Original train class balance: {np.bincount(y_train)}")

    # Apply SMOTE only on the training set (never on test set!)
    X_train_smote, y_train_smote = simple_smote(X_train_scaled, y_train)
    print(f"After SMOTE train class balance: {np.bincount(y_train_smote)}")

    models = get_models()
    trained_models = {}
    probs_dict = {}
    results = []

    for name, model in models.items():
        # KNN benefits most from SMOTE since it has no class_weight option
        if name == "KNN":
            model.fit(X_train_smote, y_train_smote)
        else:
            model.fit(X_train_scaled, y_train)

        trained_models[name] = model
        probs = model.predict_proba(X_test_scaled)[:, 1]
        probs_dict[name] = probs
        results.append(evaluate_model(name, y_test, probs))
        print(f"Trained: {name}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_results.csv", index=False)

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    print("\nSaving plots...")
    plot_roc_pr_curves(probs_dict, y_test)
    plot_confusion_matrices(trained_models, X_test_scaled, y_test)

    print(f"\nDone! Check the '{OUTPUT_DIR}/' folder for results and plots.")


if __name__ == "__main__":
    main()
