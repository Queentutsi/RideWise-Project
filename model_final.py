"""
RideWise Churn Classification Pipeline (Final Version - No XGBoost)
Author: Oluwatosin Abraham
"""

# =====================================================
# 1. IMPORTS
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import optuna
from optuna.samplers import TPESampler


# =====================================================
# 2. CONFIGURATION
# =====================================================

DATA_PATH = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\riders_ml_features.csv"
SAVE_DIR = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data"
FIGURES_DIR = os.path.join(SAVE_DIR, "Figures")

THRESHOLD = 0.35
os.makedirs(FIGURES_DIR, exist_ok=True)


# =====================================================
# 3. LOAD DATA
# =====================================================

def load_data(path):
    df = pd.read_csv(path)

    target_candidates = ["churn_probability", "churn_prob"]
    target_col = next((col for col in target_candidates if col in df.columns), None)

    if target_col is None:
        raise KeyError("Target column not found.")

    X = df.drop(columns=[target_col])
    y = (df[target_col] >= THRESHOLD).astype(int)

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
# 5. MODEL DEFINITIONS (8 MODELS)
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
        "AdaBoost": AdaBoostClassifier(n_estimators=300, random_state=42)
    }


# =====================================================
# 6. MODEL EVALUATION
# =====================================================

def evaluate_models(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    results = []

    for name, model in models.items():
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
# 7. PLOTTING UTILITIES
# =====================================================

def save_plot(fig, filename):
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} – Confusion Matrix")
    save_plot(fig, f"{model_name}_confusion_matrix.png")


def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"{model_name} – ROC Curve")
    ax.legend()
    save_plot(fig, f"{model_name}_ROC_curve.png")


def plot_precision_recall(y_test, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_title(f"{model_name} – Precision–Recall Curve")
    save_plot(fig, f"{model_name}_PR_curve.png")


def plot_model_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=results_df, x="Model", y=metric, palette="viridis", ax=ax)
        ax.set_xticklabels(results_df["Model"], rotation=90)
        ax.set_title(f"Model Comparison – {metric}")
        save_plot(fig, f"model_comparison_{metric}.png")


# =====================================================
# 8. OPTIMAL THRESHOLD
# =====================================================

def find_best_threshold(y_test, y_prob):
    thresholds = np.linspace(0, 1, 200)
    best_t, best_f1 = 0.5, 0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1


# =====================================================
# 9. OPTUNA – RANDOM FOREST TUNING
# =====================================================

def objective_rf(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

    model = RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return f1_score(y_val, y_pred)


def tune_random_forest(X_train, y_train):
    X_t, X_val, y_t, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective_rf(trial, X_t, y_t, X_val, y_val), n_trials=30)

    print("\nBest Random Forest Params:", study.best_params)
    print("Best F1:", study.best_value)

    return RandomForestClassifier(
        **study.best_params, class_weight="balanced", random_state=42, n_jobs=-1
    )


# =====================================================
# 10. MAIN PIPELINE
# =====================================================

def main():

    # -----------------------------
    # Load & preprocess
    # -----------------------------
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    print("Preprocessing...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess(X, y)

    # -----------------------------
    # Train all models
    # -----------------------------
    print("Training models...")
    models = get_models()
    results = evaluate_models(models, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

    print("\nModel Comparison:")
    print(results.sort_values("F1", ascending=False))

    # -----------------------------
    # Select best model
    # -----------------------------
    best_model_name = results.loc[results["F1"].idxmax(), "Model"]
    best_model = models[best_model_name]

    print(f"\nBest Model (before tuning): {best_model_name}")

    # -----------------------------
    # SMOTE + Random Forest tuning
    # -----------------------------
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nTuning Random Forest...")
    best_rf = tune_random_forest(X_train_resampled, y_train_resampled)
    best_rf.fit(X_train_resampled, y_train_resampled)

    best_model = best_rf
    best_model_name = "Random Forest"

    print(f"\nBest Model (after tuning): {best_model_name}")

    # -----------------------------
    # Final predictions
    # -----------------------------
    y_prob = best_model.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_test, y_prob)

    print(f"\nOptimal Threshold: {best_threshold:.3f} (F1 = {best_f1:.3f})")

    y_pred = (y_prob >= best_threshold).astype(int)

    # -----------------------------
    # Plots & metrics
    # -----------------------------
    plot_confusion_matrix(confusion_matrix(y_test, y_pred), best_model_name)
    plot_roc_curve(y_test, y_prob, best_model_name)
    plot_precision_recall(y_test, y_prob, best_model_name)
    plot_model_comparison(results)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nFinal Metrics:")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.3f}")


    # =====================================================
    # MUTUAL INFORMATION (MI) FEATURE IMPORTANCE
    # =====================================================

    print("\nComputing Mutual Information (MI) feature importance...")

    from sklearn.feature_selection import mutual_info_classif

    # Compute MI scores
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

    # Build MI dataframe
    mi_df = pd.DataFrame({
        "feature": X_train.columns,
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False)

    # Save MI table
    mi_df.to_csv(os.path.join(FIGURES_DIR, "mutual_information_scores.csv"), index=False)

    # Plot MI (Top 20)
    plt.figure(figsize=(10, 6))
    plt.bar(mi_df["feature"].head(20), mi_df["mi_score"].head(20), color="purple")
    plt.xticks(rotation=45, ha="right")
    plt.title("Mutual Information (MI) — Top 20 Features")
    plt.ylabel("MI Score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "mutual_information_top20.png"), dpi=300)
    plt.close()

    print("Saved: mutual_information_scores.csv and mutual_information_top20.png")
   

if __name__ == "__main__":
    main()
