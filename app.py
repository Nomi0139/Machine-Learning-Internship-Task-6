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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("diabetes.csv")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

X, y = df.drop("Outcome", axis=1).values, df["Outcome"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

def smote(X, y, k=5):
    rng = np.random.RandomState(42)
    X_min = X[y == 1]
    n_new = (y == 0).sum() - len(X_min)
    if n_new <= 0:
        return X, y
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X_min).kneighbors(X_min)[1]
    new_pts = [X_min[i] + rng.rand() * (X_min[idx[i][rng.randint(1, k + 1)]] - X_min[i])
               for i in rng.randint(0, len(X_min), n_new)]
    return np.vstack([X, new_pts]), np.concatenate([y, np.ones(n_new)])

X_train_sm, y_train_sm = smote(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=5),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=200),
    "SVM": SVC(class_weight="balanced", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=9),
}

results, probs_dict = [], {}

for name, model in models.items():
    model.fit(*(X_train_sm, y_train_sm) if name == "KNN" else (X_train, y_train))
    probs = model.predict_proba(X_test)[:, 1]
    probs_dict[name] = probs

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_test, probs >= t) for t in thresholds]
    best_t, best_f1 = thresholds[np.argmax(f1s)], max(f1s)

    results.append({
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_test, probs), 3),
        "PR-AUC": round(average_precision_score(y_test, probs), 3),
        "F1 (0.5)": round(f1_score(y_test, probs >= 0.5), 3),
        "Best F1": round(best_f1, 3),
        "Best Threshold": round(best_t, 2),
    })

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv("outputs/model_results.csv", index=False)
print(results_df.to_string(index=False))

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
for name, probs in probs_dict.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
for name, probs in probs_dict.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    plt.plot(rec, prec, label=name)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curves"); plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/roc_pr_curves.png", dpi=150)
