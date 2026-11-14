import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

pipe = joblib.load("model/model.joblib")

df = pd.read_csv("data/diabetes.csv")
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
X = df.drop(columns="Outcome")

imputer = pipe.named_steps["impute"]
scaler = pipe.named_steps["scale"]
clf = pipe.named_steps["clf"]

# For logistic regression, each feature's push on the log-odds is simply
# (scaled value) x (coefficient). This is what SHAP computes for linear models.
Z = scaler.transform(imputer.transform(X))
contrib = Z * clf.coef_[0]

# ---------- Global view: which features matter most overall ----------
importance = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns)
importance = importance.sort_values(ascending=True)
print("Average impact of each feature:")
print(importance.sort_values(ascending=False).round(3))

plt.figure(figsize=(7, 4))
plt.barh(importance.index, importance.values, color="#20A39E")
plt.xlabel("Average impact on risk score")
plt.title("What drives diabetes risk in this model")
plt.tight_layout()
plt.savefig("model/feature_importance.png", dpi=150)

# ---------- Local view: why one patient got their score ----------
i = 0
prob = pipe.predict_proba(X.iloc[[i]])[0, 1]
print(f"\nPatient {i}: risk probability {prob:.2f}")
print(pd.Series(contrib[i], index=X.columns)
      .sort_values(key=abs, ascending=False).round(2))
print("\n(positive = pushes risk up, negative = pushes risk down)")

print("\nSaved model/feature_importance.png")