import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. LOAD MODEL
# =========================================================

model = joblib.load(
    "models/embedding_classifier.pkl"
)


# =========================================================
# 2. LOAD EMBEDDING MODEL
# =========================================================

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# 3. LOAD UNSEEN DATA
# =========================================================

df = pd.read_csv(
    "data/evaluation.csv"
)

X = df["prompt"].tolist()
y = df["label"]


# =========================================================
# 4. CREATE EMBEDDINGS
# =========================================================

print("Creating embeddings...")

X_embeddings = embedding_model.encode(
    X,
    show_progress_bar=True
)


# =========================================================
# 5. GET INJECTION PROBABILITIES
# =========================================================

probabilities = model.predict_proba(
    X_embeddings
)

classes = model.classes_

injection_index = list(classes).index(
    "prompt_injection"
)

injection_probabilities = probabilities[
    :, injection_index
]


# =========================================================
# 6. TEST MULTIPLE THRESHOLDS
# =========================================================

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


print("\n" + "=" * 75)
print("THRESHOLD ANALYSIS")
print("=" * 75)


for threshold in thresholds:

    predictions = [
        "prompt_injection"
        if probability >= threshold
        else "benign"
        for probability in injection_probabilities
    ]

    precision = precision_score(
        y,
        predictions,
        pos_label="prompt_injection",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        pos_label="prompt_injection",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        pos_label="prompt_injection",
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[
            "benign",
            "prompt_injection"
        ]
    )

    tn, fp, fn, tp = cm.ravel()

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall:    {recall:.3f}"
    )

    print(
        f"F1:        {f1:.3f}"
    )

    print(
        f"False Positives: {fp}"
    )

    print(
        f"False Negatives: {fn}"
    )