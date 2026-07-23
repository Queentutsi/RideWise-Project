# TO TOSIN 
# feature_engineering.py → creates riders_ml_features.csv
# modelling_pipeline.py → consumes riders_ml_features.csv


# model_training.py
"""
 RideWise – Churn Modelling Pipeline
Trains 10 models step-by-step and saves charts/tables as PNG files.

Models:
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. ExtraTrees Regressor
6. Gradient Boosting Regressor
7. XGBoost Regressor
8. KNN Regressor
9. SVR Regressor
10. Logistic Regression (classification)

Outputs saved as PNG:
- MSE comparison chart
- R² comparison chart
- Regression metrics table
- Classification metrics table
"""

# =========================
# 0. Imports and data setup
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

# 1. Load engineered dataset
df = pd.read_csv(r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\riders_ml_features.csv")

# 2. Define target and features
target_candidates = ["churn_probability", "churn_prob"]
target_col = next((col for col in target_candidates if col in df.columns), None)

if target_col is None:
    raise KeyError(
        "No target column found. Expected one of: " + ", ".join(target_candidates)
    )

X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scaling (for models that need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data loaded and split successfully.")


# =========================
# 1. Linear Regression
# =========================

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)


# =========================
# 2. Ridge Regression
# =========================

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_reg.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)


# =========================
# 3. Decision Tree Regressor
# =========================

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)


# =========================
# 4. Random Forest Regressor
# =========================

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)


# =========================
# 5. ExtraTrees Regressor
# =========================

from sklearn.ensemble import ExtraTreesRegressor

et_reg = ExtraTreesRegressor(n_estimators=200, random_state=42)
et_reg.fit(X_train, y_train)
y_pred_et = et_reg.predict(X_test)

mse_et = mean_squared_error(y_test, y_pred_et)
r2_et = r2_score(y_test, y_pred_et)


# =========================
# 6. Gradient Boosting Regressor
# =========================

from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train, y_train)
y_pred_gb = gb_reg.predict(X_test)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)


# =========================
# 7. XGBoost Regressor
# =========================

from xgboost import XGBRegressor

xgb_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
xgb_reg.fit(X_train, y_train)
y_pred_xgb = xgb_reg.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)


# =========================
# 8. KNN Regressor
# =========================

from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_scaled, y_train)
y_pred_knn = knn_reg.predict(X_test_scaled)

mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)


# =========================
# 9. Support Vector Regressor (SVR)
# =========================

from sklearn.svm import SVR

svr_reg = SVR(kernel="rbf")
svr_reg.fit(X_train_scaled, y_train)
y_pred_svr = svr_reg.predict(X_test_scaled)

mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)


# =========================
# 10. Logistic Regression (classification)
# =========================

from sklearn.linear_model import LogisticRegression

threshold = 0.5
y_train_label = (y_train >= threshold).astype(int)
y_test_label = (y_test >= threshold).astype(int)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train_label)
y_pred_log_label = log_reg.predict(X_test_scaled)
y_pred_log_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

acc_log = accuracy_score(y_test_label, y_pred_log_label)
prec_log = precision_score(y_test_label, y_pred_log_label)
rec_log = recall_score(y_test_label, y_pred_log_label)
f1_log = f1_score(y_test_label, y_pred_log_label)
auc_log = roc_auc_score(y_test_label, y_pred_log_prob)


# =========================
# 11. Regression results table
# =========================

results_reg = pd.DataFrame([
    ["Linear Regression", mse_lin, r2_lin],
    ["Ridge Regression", mse_ridge, r2_ridge],
    ["Decision Tree", mse_tree, r2_tree],
    ["Random Forest", mse_rf, r2_rf],
    ["ExtraTrees", mse_et, r2_et],
    ["Gradient Boosting", mse_gb, r2_gb],
    ["XGBoost", mse_xgb, r2_xgb],
    ["KNN", mse_knn, r2_knn],
    ["SVR", mse_svr, r2_svr],
], columns=["Model", "MSE", "R2"])

print("\nRegression Model Comparison:")
print(results_reg)


# =========================
# 12. Save regression charts as PNG
# =========================

# MSE chart
plt.figure(figsize=(9, 5))
plt.bar(results_reg["Model"], results_reg["MSE"], color="teal")
plt.title("Model Comparison (MSE)")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\MSE_comparison.png",
    dpi=300, bbox_inches="tight"
)
plt.show()

# R² chart
plt.figure(figsize=(9, 5))
plt.bar(results_reg["Model"], results_reg["R2"], color="blue")
plt.title("Model Comparison (R²)")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\R2_comparison.png",
    dpi=300, bbox_inches="tight"
)
plt.show()


# =========================
# 13. Save regression table as PNG
# =========================

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=results_reg.values,
    colLabels=results_reg.columns,
    cellLoc='center',
    loc='center'
)

plt.savefig(
    r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\regression_metrics.png",
    dpi=300, bbox_inches="tight"
)
plt.show()


# =========================
# 14. Save classification table as PNG
# =========================

results_clf = pd.DataFrame([
    ["Logistic Regression", acc_log, prec_log, rec_log, f1_log, auc_log]
], columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=results_clf.values,
    colLabels=results_clf.columns,
    cellLoc='center',
    loc='center'
)

plt.savefig(
    r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\Ridewise Project\Data\classification_metrics.png",
    dpi=300, bbox_inches="tight"
)
plt.show()

print("\nAll charts and tables saved successfully.")
print("Model training complete.")
