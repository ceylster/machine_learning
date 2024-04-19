# CatBoost

# Kategorik degiskenler ile otomatik olarak mucadele edebilen, hizli, basarili bir diger GBM turevi.

import warnings as warning
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from catboost import CatBoostClassifier

df = pd.read_csv('Miull/Projects_doc/diabetes.csv')

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7735251676428148
cv_results["test_f1"].mean()
# 0.6502723851348231
cv_results["test_roc_auc"].mean()
# 0.8378923829489867
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

catbost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7721755368814192
cv_results["test_f1"].mean()
# 0.6322580676028952
cv_results["test_roc_auc"].mean()
# 0.842001397624039