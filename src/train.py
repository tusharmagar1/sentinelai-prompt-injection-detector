import os
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("data/prompts.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Total examples: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# =========================================================
# 2. INPUT AND TARGET
# =========================================================

X = df["prompt"]
y = df["label"]


# =========================================================
# 3. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")


# =========================================================
# 4. CREATE ML PIPELINE
# =========================================================

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# =========================================================
# 5. DEFINE HYPERPARAMETERS
# =========================================================

parameter_grid = {

    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],

    "tfidf__min_df": [
        1,
        2
    ],

    "classifier__C": [
        0.1,
        1,
        10
    ]
}


# =========================================================
# 6. CROSS-VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# 7. F1 SCORER
# =========================================================

f1_injection_scorer = make_scorer(
    f1_score,
    pos_label="prompt_injection"
)


# =========================================================
# 8. GRID SEARCH
# =========================================================

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    cv=cv,
    scoring=f1_injection_scorer,
    n_jobs=-1
)


print("\n" + "=" * 60)
print("TRAINING + HYPERPARAMETER TUNING")
print("=" * 60)

print("Testing multiple model configurations...")

grid_search.fit(
    X_train,
    y_train
)


# =========================================================
# 9. GET BEST MODEL
# =========================================================

best_model = grid_search.best_estimator_

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Parameters:")
print(grid_search.best_params_)

print(
    "\nBest Cross-Validation F1 Score:"
)

print(
    f"{grid_search.best_score_:.4f}"
)


# =========================================================
# 10. SAVE TRAINED MODEL
# =========================================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = (
    "models/prompt_injection_model.pkl"
)

joblib.dump(
    best_model,
    model_path
)

print("\nModel saved successfully!")
print(f"Location: {model_path}")


# =========================================================
# 11. FINAL TEST PREDICTIONS
# =========================================================

y_pred = best_model.predict(
    X_test
)


# =========================================================
# 12. TEST ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# 13. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 14. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "benign",
        "prompt_injection"
    ]
)

print("\nConfusion Matrix:")

print(cm)


# =========================================================
# 15. DISPLAY CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=[
        "Benign",
        "Prompt Injection"
    ],
    yticklabels=[
        "Benign",
        "Prompt Injection"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Prompt Injection Detection - Confusion Matrix"
)

plt.tight_layout()

plt.show()