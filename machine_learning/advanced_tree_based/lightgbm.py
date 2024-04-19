# LightGBM

# LightGBM, XGBoost'un egitim suresi, performansini arttirmaya yonelik gelistirilen bir diger GBM turudur.
# LightGBMİN en onemli hiperparametre sayisi tahmin sayisidir. Tahmin sayisi iterasyon sayisina esittir.
# Level-wise buyume stratejisi yerine Leaf-wise buyume stratejisi ile daha hizlidir.
# LightGBM'in basarili olmasinin sebebi split etme yani dallara ayirma, bolme, buyume, yontemindeki farkliliktir.
# XGbboost degiskenleri bolme islemi soz konusu oldugunda bu noktada level-wise yani seviyeye gore buyume yontemi izler,
# LightGBM bolme noktalara, yapraklara odaklaniyor. Agac yapilarindaki bolme islemleri dusunuldugunde XGBoost genis
# kapsamli bir ilk arama yaparken LightGBM derinlemesine ilk arama yapmaktadir.

import warnings as warning
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from lightgbm import LGBMClassifier

df = pd.read_csv('Miull/Projects_doc/diabetes.csv')

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7474492827434004
cv_results["test_f1"].mean()
# 0.624110522144179
cv_results["test_roc_auc"].mean()
# 0.7990293501048218
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
# 0.7643578643578645
cv_results["test_f1"].mean()
# 0.6372062920577772
cv_results["test_roc_auc"].mean()
# 0.8147491264849755

# LightGBMde temel olarak diger parametreleri belirledikten sonra, buradaki tahminci sayisini, n_estimators sayisini
# 10000lere kadar denemeliyiz.