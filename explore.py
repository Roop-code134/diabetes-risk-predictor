import pandas as pd

df = pd.read_csv("data/diabetes.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))

print("\nOutcome counts (0 = no diabetes, 1 = diabetes):")
print(df["Outcome"].value_counts())

cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nZeros in columns where zero is impossible:")
print((df[cols] == 0).sum())