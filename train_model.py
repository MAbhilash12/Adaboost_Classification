import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

cancer = load_breast_cancer(as_frame=True)

df = cancer.frame

# Save dataset
df.to_csv("data/breast_cancer.csv", index=False)

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop("target", axis=1)

y = df["target"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL
# ==========================================

base_model = DecisionTreeClassifier(
    max_depth=1
)

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy : {accuracy:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/adaboost_classifier.pkl"
)

print("\nModel Saved Successfully!")