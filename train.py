import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Load data; zeros that are impossible become "missing" ----------
df = pd.read_csv("data/diabetes.csv")
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

X = df.drop(columns="Outcome")
y = df["Outcome"]

# Hold back 20% of patients for a final, honest test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------- Three candidate models ----------
# The median fill lives inside each pipeline, so it is learned from
# training data only (no leaking of test data).
def make(clf, scale=False):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)

models = {
    "Logistic Regression": make(
        LogisticRegression(max_iter=1000, class_weight="balanced"), scale=True),
    "Random Forest": make(
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=3, random_state=42)),
    "Gradient Boosting": make(
        GradientBoostingClassifier(random_state=42)),
}

# ---------- Compare them with 5-fold cross-validation ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Cross-validation on the training set:")
results = {}
for name, pipe in models.items():
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=["roc_auc", "recall"])
    auc = scores["test_roc_auc"].mean()
    rec = scores["test_recall"].mean()
    results[name] = auc
    print(f"  {name:20s} ROC-AUC {auc:.3f} | recall {rec:.3f}")

best_name = max(results, key=results.get)
best = models[best_name]
print(f"\nBest model: {best_name}")

# ---------- Pick a threshold that catches at least 80% of diabetic patients ----------
# Done on the training data only, so the test set stays untouched.
probs = cross_val_predict(best, X_train, y_train, cv=cv,
                          method="predict_proba")[:, 1]
threshold = 0.5
for t in np.arange(0.90, 0.05, -0.01):
    if recall_score(y_train, probs >= t) >= 0.80:
        threshold = round(float(t), 2)
        break
print(f"Chosen threshold: {threshold} (default would be 0.50)")

# ---------- Final test on patients the model has never seen ----------
best.fit(X_train, y_train)
test_probs = best.predict_proba(X_test)[:, 1]
test_pred = (test_probs >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
metrics = {
    "model": best_name,
    "threshold": threshold,
    "roc_auc": round(roc_auc_score(y_test, test_probs), 3),
    "accuracy": round(accuracy_score(y_test, test_pred), 3),
    "recall": round(recall_score(y_test, test_pred), 3),
    "precision": round(precision_score(y_test, test_pred), 3),
}
print("\nTest set results:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print(f"\nMissed diabetic patients: {fn} of {fn + tp}")
print(f"False alarms: {fp} of {fp + tn} healthy patients")

# ---------- Save everything the API will need ----------
os.makedirs("model", exist_ok=True)
joblib.dump(best, "model/model.joblib")
with open("model/info.json", "w") as f:
    json.dump({**metrics, "features": list(X.columns)}, f, indent=2)
print("\nSaved model/model.joblib and model/info.json")