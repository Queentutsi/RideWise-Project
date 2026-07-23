"""
RideWise Churn Classification Pipeline
Author: Oluwatosin
"""

# =====================================================
# 1. IMPORTS
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# =====================================================
# 2. CONFIGURATION
# =====================================================

DATA_PATH = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\riders_ml_features.csv"

SAVE_DIR = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data"

THRESHOLD = 0.35   # Better recall for churn


# =====================================================
# 3. LOAD DATA
# =====================================================

def load_data(path):
    df = pd.read_csv(path)

    target_candidates = ["churn_probability", "churn_prob"]
    target_col = next((col for col in target_candidates if col in df.columns), None)

    if target_col is None:
        raise KeyError(f"No target column found. Expected one of: {target_candidates}")

    X = df.drop(columns=[target_col])
    y_prob = df[target_col]

    # Convert probability → binary churn label
    y = (y_prob >= THRESHOLD).astype(int)

    return X, y


# =====================================================
# 4. PREPROCESSING
# =====================================================

def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


# =====================================================
# 5. MODEL DEFINITIONS
# =====================================================

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVC (RBF)": SVC(kernel="rbf", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
        "LightGBM": LGBMClassifier(random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }


# =====================================================
# 6. EVALUATE MODELS
# =====================================================

def evaluate_models(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    results = []

    for name, model in models.items():

        # Use scaled data for models that need it
        if name in ["Logistic Regression", "KNN", "SVC (RBF)"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        results.append([
            name,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0),
            roc_auc_score(y_test, y_prob)
        ])

    return pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])


# =====================================================
# 7. FIGURE GENERATION
# =====================================================

def save_plot(fig, filename):
    fig.savefig(os.path.join(SAVE_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} 2 – Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_plot(fig, f"{model_name}_2_confusion_matrix.png")


def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC)")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"{model_name} 2 – ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    save_plot(fig, f"{model_name}_2_ROC_curve.png")


def plot_precision_recall(y_test, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_title(f"{model_name} 2 – Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    save_plot(fig, f"{model_name}_2_PR_curve.png")


def plot_model_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=results_df, x="Model", y=metric, palette="viridis", ax=ax)
        ax.set_xticklabels(results_df["Model"], rotation=60)
        ax.set_title(f"Model Comparison – {metric}")
        save_plot(fig, f"model_comparison_{metric}.png")


# =====================================================
# 8. MAIN PIPELINE
# =====================================================

def main():
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    print("Preprocessing...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess(X, y)

    print("Training models...")
    models = get_models()
    results = evaluate_models(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

    print("\nModel Comparison:")
    print(results.sort_values("AUC", ascending=False))

    # Select best model
    best_model_name = results.sort_values("AUC", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    # Predictions
    if best_model_name in ["Logistic Regression", "KNN", "SVC (RBF)"]:
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

    # Generate figures
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, best_model_name)
    plot_roc_curve(y_test, y_prob, best_model_name)
    plot_precision_recall(y_test, y_prob, best_model_name)
    plot_model_comparison(results)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
