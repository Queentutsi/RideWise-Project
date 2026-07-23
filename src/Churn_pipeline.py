"""
RideWise Churn Prediction Pipeline
Author: Oluwatosin
Description:
    Clean modular ML pipeline for churn classification.
"""

# =====================================================
# 1. IMPORTS
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# =====================================================
# 2. CONFIGURATION
# =====================================================

DATA_PATH = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\riders_ml_features.csv"
THRESHOLD = 0.35   # Lower threshold improves churn detection


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
# 4. TRAIN/TEST SPLIT + SCALING
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
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVC (RBF)": SVC(kernel="rbf", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    }


# =====================================================
# 6. TRAIN + EVALUATE MODELS
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
# 7. CONFUSION MATRIX FOR BEST MODEL
# =====================================================

def plot_confusion_matrix(model, X_test, X_test_scaled, y_test, model_name):
    if model_name in ["Logistic Regression", "KNN", "SVC (RBF)"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    plt.matshow(cm, cmap="Blues", alpha=0.7)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


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

    # =========================
    # Select best model
    # =========================
    best_model_name = results.sort_values("AUC", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    # Plot confusion matrix
    plot_confusion_matrix(best_model, X_test, X_test_scaled, y_test, best_model_name)

    # Classification report
    print("\nClassification Report:")
    if best_model_name in ["Logistic Regression", "KNN", "SVC (RBF)"]:
        y_pred = best_model.predict(X_test_scaled)
    else:
        y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()

# =====================================================
# 9. FIGURE GENERATION MODULES
# =====================================================

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=300)
    plt.show()


def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC)")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_ROC_curve.png", dpi=300)
    plt.show()


def plot_precision_recall(y_test, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=model_name)
    plt.title(f"Precision–Recall Curve - {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"{model_name}_PR_curve.png", dpi=300)
    plt.show()


def plot_model_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    for metric in metrics:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results_df, x="Model", y=metric, palette="viridis")
        plt.xticks(rotation=60)
        plt.title(f"Model Comparison – {metric}")
        plt.tight_layout()
        plt.savefig(f"model_comparison_{metric}.png", dpi=300)
        plt.show()


def save_metrics_table(results_df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        cellLoc="center",
        loc="center"
    )
    plt.savefig("classification_metrics_table.png", dpi=300)
    plt.show()
