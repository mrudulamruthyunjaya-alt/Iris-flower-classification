"""
Task 1: Iris Flower Classification
Train ML models to classify Iris species from flower measurements.
Usage: python iris_classification.py [path/to/Iris.csv]
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ------------------------------------------------------------------ 1. Load
path = sys.argv[1] if len(sys.argv) > 1 else "Iris.csv"
df = pd.read_csv(path)
df = df.drop(columns=["Id"], errors="ignore")          # Id is just a row index
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("\nClass balance:\n", df["Species"].value_counts())
print("\nSummary statistics:\n", df.describe().round(2))

# ------------------------------------------------------------------ 2. Explore
sns.pairplot(df, hue="Species", corner=True)
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------ 3. Split
X = df.drop(columns="Species")
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ------------------------------------------------------------------ 4. Models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    "KNN (k=5)":           make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC()),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)   # train data only
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    rows.append((name, cv_scores.mean(), cv_scores.std(), test_acc))

results = pd.DataFrame(rows, columns=["Model", "CV Acc (mean)", "CV Acc (std)", "Test Acc"])
results = results.sort_values("Test Acc", ascending=False).reset_index(drop=True)
print("\nModel comparison:\n", results.round(4).to_string(index=False))

# ------------------------------------------------------------------ 5. Evaluate best
best_name = results.loc[0, "Model"]
best = models[best_name]
y_pred = best.predict(X_test)
print(f"\n=== Best model: {best_name} ===")
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:\n", classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=best.classes_),
                       display_labels=best.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {best_name}")
plt.savefig("iris_confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------------ 6. Predict new flower
sample = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]], columns=X.columns)
print("\nPrediction for sample [5.9, 3.0, 5.1, 1.8]:", best.predict(sample)[0])
