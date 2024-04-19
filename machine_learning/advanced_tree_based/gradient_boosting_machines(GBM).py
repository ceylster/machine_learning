# Gradient Boosting Machine (GBM)
# GBM artik optimizasyonuna dayali calisan bir agac yontemidir. Agac yontemlerine, boosting ve gradient descent
# yontemlerinin uygulanmasidir. Agaclar birbirlerine bagimlidir.
# AdaBoost (Adaptive Boosting): Zayif siniflandiricinin bir araya gelerek guclu bir siniflandirici olusturmasi fikrine
# dayanir. Gbmin temelini olusturur.
# Gradient boosting tek bir tahminsel model formunda olan modeller serisi olusturur.
# Hatalar/artiklar uzerine tek bir tahminsel model formunda olan modeller serisi kurulur.
# GBM diferansiyellenebilen herhangi bir kayip fonksiyonunu optimize edebilen Gradient Descent algoritmasini
# kullanmaktadir.
# Tek bir tahminsel model formunda olan modeller serisi additive sekilde kurulur.
from sklearn.ensemble import GradientBoostingClassifier

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

# 'learning_rate' artik tahminlerin guncellenme hizi, 'max_depth', 'max_features', 'min_samples_split',
# 'n_estimators' burada optimizasyon sayisidir.

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

# learning rate ne kadar kucuk olursa train suresi o kadar uzamaktadir. Ancak kucuk olmasi durumunda daha basarili
# tahminler elde  edilmektedir.

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state= 17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()