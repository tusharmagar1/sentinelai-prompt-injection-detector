import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    "models/prompt_injection_model.pkl"
)


# =========================================================
# LOAD UNSEEN DATASET
# =========================================================

df = pd.read_csv(
    "data/evaluation.csv"
)

X = df["prompt"]
y = df["label"]


# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(X)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y,
    predictions
)

print("=" * 60)
print("UNSEEN EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y,
        predictions
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y,
    predictions,
    labels=[
        "benign",
        "prompt_injection"
    ]
)

print("\nConfusion Matrix:")

print(cm)