import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv("heart_disease_uci.csv")
print("=" * 60)
print("HEART DISEASE PREDICTION USING SVM")
print("=" * 60)
print("\nOriginal Dataset Shape:")
print(df.shape)

# ============================================================
# 2. REMOVE UNNECESSARY COLUMNS
# ============================================================
df = df.drop(columns=["id", "dataset"])
print("\nAfter Removing id and dataset:")
print(df.shape)

# ============================================================
# 3. CONVERT TARGET INTO BINARY CLASS
# ============================================================
df["num"] = (df["num"] > 0).astype(int)
print("\nTarget Distribution:")
print(df["num"].value_counts().sort_index())
print("\nTarget Meaning:")
print("0 = No Disease")
print("1 = Disease")

# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================
X = df.drop(columns=["num"])
y = df["num"]

# ============================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================
numerical_features = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

print("\nNumerical Features:")
print(numerical_features)
print("\nCategorical Features:")
print(categorical_features)

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ============================================================
# 7. PREPROCESSING
# ============================================================
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# ============================================================
# 8. DEFINE SVM MODELS
# ============================================================
models = {
    "Linear SVM": SVC(kernel="linear", random_state=42),
    "RBF SVM": SVC(kernel="rbf", random_state=42),
    "Polynomial SVM": SVC(kernel="poly", degree=3, random_state=42)
}

# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================
results = {}
confusion_matrices = {}

for model_name, model in models.items():
    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
    confusion_matrices[model_name] = cm

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Disease", "Disease"],
        zero_division=0
    ))

# ============================================================
# 10. MODEL COMPARISON
# ============================================================
results_df = pd.DataFrame(results).T
print("\n")
print("=" * 60)
print("SVM KERNEL COMPARISON")
print("=" * 60)
print(results_df.round(4))

# ============================================================
# 11. FIND BEST BASELINE MODEL
# ============================================================
best_model = results_df["F1-Score"].idxmax()
print("\nBest Performing Baseline Model:")
print(best_model)
print("\nBest Baseline Model Metrics:")
print(results_df.loc[best_model].round(4))

# ============================================================
# 12. 5-FOLD CROSS-VALIDATION
# ============================================================
print("\n")
print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}

cv_results = {}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    # CV uses training data only; test data stays untouched for final evaluation.
    scores = cross_validate(
        pipeline, X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=1
    )

    cv_results[model_name] = {
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1-Score": scores["test_f1"].mean()
    }

cv_df = pd.DataFrame(cv_results).T
print("\nAverage 5-Fold Cross-Validation Results:")
print(cv_df.round(4))

best_cv_model = cv_df["F1-Score"].idxmax()
print("\nBest Kernel Based on Cross-Validation F1-Score:")
print(best_cv_model)

# ============================================================
# 13. RBF SVM HYPERPARAMETER TUNING
# ============================================================
print("\n")
print("=" * 60)
print("RBF SVM HYPERPARAMETER TUNING")
print("=" * 60)

rbf_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", SVC(kernel="rbf"))
])

param_grid = {
    "classifier__C": [0.1, 1, 10, 100],
    "classifier__gamma": ["scale", 0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(
    estimator=rbf_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=1
)

print("\nSearching for the best C and gamma values...")
grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)
print("\nBest Cross-Validation F1-Score:")
print(round(grid_search.best_score_, 4))

# ============================================================
# 14. EVALUATE TUNED RBF MODEL ON TEST DATA
# ============================================================
best_rbf_model = grid_search.best_estimator_
y_pred_tuned = best_rbf_model.predict(X_test)

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_precision = precision_score(y_test, y_pred_tuned, zero_division=0)
tuned_recall = recall_score(y_test, y_pred_tuned, zero_division=0)
tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
tuned_cm = confusion_matrix(y_test, y_pred_tuned)

print("\n")
print("=" * 60)
print("TUNED RBF SVM TEST RESULTS")
print("=" * 60)
print("\nAccuracy :", round(tuned_accuracy, 4))
print("Precision:", round(tuned_precision, 4))
print("Recall   :", round(tuned_recall, 4))
print("F1-Score :", round(tuned_f1, 4))
print("\nConfusion Matrix:")
print(tuned_cm)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_tuned,
    target_names=["No Disease", "Disease"],
    zero_division=0
))

# ============================================================
# 15. COMPARE BASELINE RBF WITH TUNED RBF
# ============================================================
print("\n")
print("=" * 60)
print("BASELINE RBF VS TUNED RBF")
print("=" * 60)

comparison = pd.DataFrame({
    "Baseline RBF": [
        results_df.loc["RBF SVM", "Accuracy"],
        results_df.loc["RBF SVM", "Precision"],
        results_df.loc["RBF SVM", "Recall"],
        results_df.loc["RBF SVM", "F1-Score"]
    ],
    "Tuned RBF": [
        tuned_accuracy,
        tuned_precision,
        tuned_recall,
        tuned_f1
    ]
}, index=["Accuracy", "Precision", "Recall", "F1-Score"])

print(comparison.round(4))

# ============================================================
# 16. FINAL MODEL
# ============================================================
final_model = best_rbf_model

print("\n")
print("=" * 60)
print("FINAL MODEL")
print("=" * 60)
print("Final Model: Tuned RBF SVM")
print("Best Parameters:", grid_search.best_params_)
print("Test Accuracy :", round(tuned_accuracy, 4))
print("Test Precision:", round(tuned_precision, 4))
print("Test Recall   :", round(tuned_recall, 4))
print("Test F1-Score :", round(tuned_f1, 4))

# ============================================================
# SAVE FINAL MODEL
# ============================================================

joblib.dump(final_model, "heart_disease_model.pkl")

print("\nFinal model saved successfully!")
print("Saved as: heart_disease_model.pkl")

# ============================================================
# 17. VISUALIZE MODEL PERFORMANCE
# ============================================================

ax = results_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("SVM Kernel Performance Comparison")
plt.xlabel("SVM Kernel")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Evaluation Metrics")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.tight_layout()


# ============================================================
# 18. BASELINE RBF VS TUNED RBF GRAPH
# ============================================================

ax = comparison.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Baseline RBF vs Tuned RBF")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")

# Keep 0 to 1 scale
plt.ylim(0, 1)

plt.xticks(rotation=0)

# Legend below the graph
plt.legend(
    title="Model",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=2
)

# Display values above bars
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3
    )

plt.tight_layout()


# ============================================================
# 19. CONFUSION MATRIX FOR FINAL MODEL
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(tuned_cm)

plt.title("Confusion Matrix - Tuned RBF SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["No Disease", "Disease"]
)

plt.yticks(
    [0, 1],
    ["No Disease", "Disease"]
)

# Display values inside the matrix
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            tuned_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()


# ============================================================
# 20. SHOW ALL GRAPHS
# ============================================================

plt.show()


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 60)
print("PROJECT EXECUTION COMPLETED")
print("=" * 60)

#python predict.py