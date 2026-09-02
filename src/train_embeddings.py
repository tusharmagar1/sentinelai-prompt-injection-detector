import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv(
    "data/prompts.csv"
)

X = df["prompt"].tolist()
y = df["label"]


print("=" * 60)
print("EMBEDDING MODEL")
print("=" * 60)

print(
    f"Total examples: {len(X)}"
)


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(
    f"\nTraining examples: {len(X_train)}"
)

print(
    f"Testing examples: {len(X_test)}"
)


# =========================================================
# 3. LOAD SENTENCE TRANSFORMER
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# 4. CREATE EMBEDDINGS
# =========================================================

print("Creating training embeddings...")

X_train_embeddings = embedding_model.encode(
    X_train,
    show_progress_bar=True
)


print("\nCreating test embeddings...")

X_test_embeddings = embedding_model.encode(
    X_test,
    show_progress_bar=True
)


print(
    "\nTraining embedding shape:",
    X_train_embeddings.shape
)

print(
    "Testing embedding shape:",
    X_test_embeddings.shape
)


# =========================================================
# 5. LOGISTIC REGRESSION
# =========================================================

classifier = LogisticRegression(
    max_iter=1000
)


# =========================================================
# 6. HYPERPARAMETER SEARCH
# =========================================================

parameter_grid = {
    "C": [
        0.1,
        1,
        10
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameter_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)


print("\nTraining classifier...")

grid_search.fit(
    X_train_embeddings,
    y_train
)


# =========================================================
# 7. BEST MODEL
# =========================================================

best_model = grid_search.best_estimator_


print(
    "\nBest parameters:",
    grid_search.best_params_
)


print(
    "Best CV F1:",
    f"{grid_search.best_score_ * 100:.2f}%"
)


# =========================================================
# 8. TEST PREDICTIONS
# =========================================================

y_pred = best_model.predict(
    X_test_embeddings
)


# =========================================================
# 9. TEST ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# 10. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# 11. CONFUSION MATRIX
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
# 12. SAVE MODEL
# =========================================================

joblib.dump(
    best_model,
    "models/embedding_classifier.pkl"
)


print(
    "\nClassifier saved successfully!"
)