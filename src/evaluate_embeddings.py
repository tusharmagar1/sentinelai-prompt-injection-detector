import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# LOAD CLASSIFIER
# =========================================================

classifier = joblib.load(
    "models/embedding_classifier.pkl"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# LOAD UNSEEN DATASET
# =========================================================

df = pd.read_csv(
    "data/evaluation.csv"
)

X = df["prompt"].tolist()
y = df["label"]


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print("Creating evaluation embeddings...")

X_embeddings = embedding_model.encode(
    X,
    show_progress_bar=True
)


# =========================================================
# PREDICTIONS
# =========================================================

predictions = classifier.predict(
    X_embeddings
)


# =========================================================
# RESULTS
# =========================================================

accuracy = accuracy_score(
    y,
    predictions
)


print("\n" + "=" * 60)
print("UNSEEN EMBEDDING MODEL EVALUATION")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


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