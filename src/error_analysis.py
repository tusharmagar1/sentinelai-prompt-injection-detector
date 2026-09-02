import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# =========================================================
# 1. LOAD MODEL
# =========================================================

model = joblib.load(
    "models/prompt_injection_model.pkl"
)


# =========================================================
# 2. LOAD EVALUATION DATASET
# =========================================================

df = pd.read_csv(
    "data/evaluation.csv"
)


X = df["prompt"]
y = df["label"]


# =========================================================
# 3. MAKE PREDICTIONS
# =========================================================

predictions = model.predict(X)

probabilities = model.predict_proba(X)


# =========================================================
# 4. FIND INJECTION CLASS
# =========================================================

classes = model.classes_

injection_index = list(classes).index(
    "prompt_injection"
)


# =========================================================
# 5. ADD RESULTS TO DATAFRAME
# =========================================================

df["prediction"] = predictions

df["injection_probability"] = (
    probabilities[:, injection_index]
)


df["risk_score"] = (
    df["injection_probability"] * 100
).round(2)


# =========================================================
# 6. FIND WRONG PREDICTIONS
# =========================================================

errors = df[
    df["label"] != df["prediction"]
]


print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)


print(
    f"\nTotal examples: {len(df)}"
)

print(
    f"Incorrect predictions: {len(errors)}"
)


# =========================================================
# 7. FALSE POSITIVES
# =========================================================

false_positives = df[
    (df["label"] == "benign") &
    (df["prediction"] == "prompt_injection")
]


print("\n" + "=" * 70)
print("FALSE POSITIVES")
print("=" * 70)


if len(false_positives) == 0:

    print("\nNo false positives found.")

else:

    for index, row in false_positives.iterrows():

        print("\nPrompt:")
        print(row["prompt"])

        print(
            f"Risk Score: {row['risk_score']}"
        )

        print(
            f"Actual: {row['label']}"
        )

        print(
            f"Predicted: {row['prediction']}"
        )


# =========================================================
# 8. FALSE NEGATIVES
# =========================================================

false_negatives = df[
    (df["label"] == "prompt_injection") &
    (df["prediction"] == "benign")
]


print("\n" + "=" * 70)
print("FALSE NEGATIVES")
print("=" * 70)


if len(false_negatives) == 0:

    print("\nNo false negatives found.")

else:

    for index, row in false_negatives.iterrows():

        print("\nPrompt:")
        print(row["prompt"])

        print(
            f"Risk Score: {row['risk_score']}"
        )

        print(
            f"Actual: {row['label']}"
        )

        print(
            f"Predicted: {row['prediction']}"
        )


# =========================================================
# 9. ALL INCORRECT PREDICTIONS
# =========================================================

print("\n" + "=" * 70)
print("ALL INCORRECT PREDICTIONS")
print("=" * 70)


if len(errors) == 0:

    print("\nThe model made no errors.")

else:

    print(
        errors[
            [
                "prompt",
                "label",
                "prediction",
                "risk_score"
            ]
        ].to_string(index=False)
    )


    # =========================================================
# 10. ERROR SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    f"\nFalse Positives: {len(false_positives)}"
)

print(
    f"False Negatives: {len(false_negatives)}"
)

print(
    f"Total Errors: {len(errors)}"
)