# XGBOOST (eXtreme Gradient Boosting)
# XGBoost, GBM'in hiz ve tahmin performansini arttirmak uzere optimize edilmis; olceklenebilir ve farkli platformlarda
# entegre edilebilir versiyondur.

import warnings as warning
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
df = pd.read_csv('Miull/Projects_doc/diabetes.csv')

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

xgboost_model = XGBClassifier(random_state=17)
xgboost_model.get_params()

cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

xgboost_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [5, 8, None],
                  "n_estimators": [100, 500, 1000, None],
                  "colsample_bytree": [None, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()