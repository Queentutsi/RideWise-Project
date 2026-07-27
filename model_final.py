"""
RideWise Churn Classification Pipeline (Final Version)
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

# Models (no external installations required)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler



# =====================================================
# 2. CONFIGURATION
# =====================================================

DATA_PATH = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\riders_ml_features.csv"

SAVE_DIR = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data"
FIGURES_DIR = os.path.join(SAVE_DIR, "Figures")

THRESHOLD = 0.35   # Better recall for churn

# Ensure folders exist
os.makedirs(FIGURES_DIR, exist_ok=True)


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
# 5. MODEL DEFINITIONS (10 models)
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
        "AdaBoost": AdaBoostClassifier(n_estimators=300, random_state=42),
        # "Bagging": BaggingClassifier(n_estimators=300, random_state=42)
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
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
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
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.barplot(
            data=results_df,
            x="Model",
            y=metric,
            hue="Model",
            dodge=False,
            legend=False,
            palette="viridis",
            ax=ax
        )

        ax.set_xticks(range(len(results_df["Model"])))
        ax.set_xticklabels(results_df["Model"], rotation=90, ha="right")

    
        ax.set_title(f"Model Comparison – {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Model")

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
    print(results.sort_values("F1", ascending=False))

    # =====================================================
    # Select best model using F1
    # =====================================================
    best_model_name = results.loc[results["F1"].idxmax(), "Model"]
    best_model = models[best_model_name]

    print(f"\nBest Model (before tuning): {best_model_name}")

    # =====================================================
    # Prepare resampled data for tuning
    # =====================================================
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # =====================================================
    # FORCE Random Forest tuning
    # =====================================================
    print("\nForcing Random Forest tuning...")
    best_model = tune_random_forest(X_train_resampled, y_train_resampled)
    best_model_name = "Random Forest"
    best_model.fit(X_train_resampled, y_train_resampled)

    print(f"\nBest Model (after tuning): {best_model_name}")

    # =====================================================
    # Optional: Tune XGBoost after Random Forest
    # This compares tuned RF vs tuned XGB, Automatically switches to XGB if better
# Keeps RF if RF is still best, Fully compatible with your threshold optimiser
    # =====================================================
    print("\nRunning Optuna tuning for XGBoost...")
    best_model_xgb = tune_xgboost(X_train_resampled, y_train_resampled)
    best_model_xgb.fit(X_train_resampled, y_train_resampled)

    # Compare tuned RF vs tuned XGB using F1 on validation
    y_prob_rf = best_model.predict_proba(X_test)[:, 1]
    y_prob_xgb = best_model_xgb.predict_proba(X_test)[:, 1]

    f1_rf = f1_score(y_test, (y_prob_rf >= 0.5).astype(int))
    f1_xgb = f1_score(y_test, (y_prob_xgb >= 0.5).astype(int))

    if f1_xgb > f1_rf:
        print("\nXGBoost outperformed Random Forest — switching to XGBoost.")
        best_model = best_model_xgb
        best_model_name = "XGBoost"
    else:
        print("\nRandom Forest remains the best model.")


    # =====================================================
    # Get probabilities
    # =====================================================
    if best_model_name in ["Logistic Regression", "KNN", "SVC (RBF)"]:
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = best_model.predict_proba(X_test)[:, 1]

    # =====================================================
    # Find optimal threshold
    # =====================================================
    best_threshold, best_f1 = find_best_threshold(y_test, y_prob)
    print(f"\nOptimal Threshold: {best_threshold:.3f} (F1 = {best_f1:.3f})")

    # Apply threshold
    y_pred = (y_prob >= best_threshold).astype(int)

    # =====================================================
    # Generate plots
    # =====================================================
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, best_model_name)
    plot_roc_curve(y_test, y_prob, best_model_name)
    plot_precision_recall(y_test, y_prob, best_model_name)
    plot_model_comparison(results)

    # =====================================================
    # Classification report
    # =====================================================
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# =====================================================
# 9. FIND BEST THRESHOLD - Searches 200 thresholds, Picks the one with the highest F1
# =====================================================
def find_best_threshold(y_test, y_prob):
    thresholds = np.linspace(0.0, 1.0, 200)
    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


# =====================================================
# 10. OPTUNA HYPERPARAMETER TUNING (Optional -explain as - Optuna objective function (Decision Tree tuning))
# Tunes Decision Tree, Uses F1 as the optimisation metric, Uses class_weight="balanced", Works with SMOTE, Works with your pipeline
# =====================================================
def objective_dt(trial, X_train, y_train, X_val, y_val):
    # Hyperparameters to tune
    # to tune the dt, i changed the max_depth from 3,30 to 3,15). this cuts tree building time.
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return f1_score(y_val, y_pred)

# ======================================================
# 11. TUNE DECISION TREE FUNCTION (0.5-1.0 F1 score, Uses Optuna, 
#     Returns a fully tuned Decision Tree model, 50 trials (fast but effective)
#     Uses TPE sampler (best for ML tuning))
# ======================================================
def tune_decision_tree(X_train, y_train):
    # Create validation split
    X_t, X_val, y_t, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    # there was a keyboard interrupt error here, so I chnaged the n_trial from 50 to 20, Reducing run time by 60%
    study.optimize(lambda trial: objective_dt(trial, X_t, y_t, X_val, y_val), n_trials=20)   # CHANGED HERE
    print("\nBest Decision Tree Params:", study.best_params)
    print("Best F1:", study.best_value)

    # Build final tuned model
    best_params = study.best_params

    tuned_model = DecisionTreeClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42
    )

    return tuned_model


# ======================================================
# 12. RANDOM FOREST Optuna objective function (Uses class_weight="balanced"
# Uses F1 as optimisation metric, Works with SMOTE, Uses n_jobs=-1 for speed)
# ======================================================

def objective_rf(trial, X_train, y_train, X_val, y_val):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    return f1_score(y_val, y_pred)

# ======================================================
# 13. RANDOM FOREST tuning function
# 30 trials (fast + effective), Returns a fully tuned RF model, Prints best parameters + best F1
# ======================================================

def tune_random_forest(X_train, y_train):
    # Validation split
    X_t, X_val, y_t, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )

    study.optimize(lambda trial: objective_rf(trial, X_t, y_t, X_val, y_val), n_trials=30)

    print("\nBest Random Forest Params:", study.best_params)
    print("Best F1:", study.best_value)

    best_params = study.best_params

    tuned_model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return tuned_model


# ======================================================
# 14. Optuna Objective Function for XGBoost (Uses cses realistic XGB hyperparameter ranges
# Uses F1 score, Works with SMOTE, Works with the pipeline)
# ======================================================

def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred)



# ======================================================
# 15. XGBoost Tuning Wrapper Function (30 trials (fast + effective),
#  Returns tuned XGB model, Prints best params + best F1)
# ======================================================

def tune_xgboost(X_train, y_train):
    X_t, X_val, y_t, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective_xgb(trial, X_t, y_t, X_val, y_val), n_trials=30)

    print("\nBest XGBoost Params:", study.best_params)
    print("Best F1:", study.best_value)

    best_params = study.best_params
    best_params["eval_metric"] = "logloss"
    best_params["use_label_encoder"] = False
    best_params["random_state"] = 42

    tuned_model = XGBClassifier(**best_params)
    return tuned_model

if __name__ == "__main__":
    main()


    # from model_final import main
