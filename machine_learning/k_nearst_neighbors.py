# K-Nearest Neighbors (K-En yakin komsuluk)

# K-Nearest Neighbors Regression (K-En Yakin Komsu Regresyon)
# Gozlemleri birbirine olan benzerlikleri uzerinden tahmin yapilmasi.
# Oklid ya da benzeri bir uzaklik hesabi ile her bir gozleme uzaklik hesaplanir.

# K-Nearest Neighbors Classification (K-En Yakin Komsu Siniflandirmasi)
# Gozlemleri birbirine olan benzerlikleri uzerinden tahmin yapilmasi.
# En yakin K adet gozlemin y degerinin en sik gozlenen frekansi tahmin edilen sinif olur.

import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# 1. Exploratory Data Analysis

df = pd.read_csv("Miull/Projects_doc/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

# 2. Data Pre-processing(Veri On Isleme)

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Uzaklik temelli yontemlerde ve Gradient discent temelli yontemlerde degiskenlerin standart olmasi elde edilecek
# sonuclarin ya daha hizli ya da daha basarili olmasini saglayacaktir. Elimizdeki bagimsiz degiskenleri standartlastirma
# islemine sokariz.
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 3. Modelling & Prediction

knn_model = KNeighborsClassifier().fit(X,y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

# Model Evaluation

# Confusion matrix icin y_pred:
y_pred = knn_model.predict(X)

# AUC icin y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74

# AUC
roc_auc_score(y, y_prob)
# 0.9017686567164179

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# birden fazla metrige gore degerlendirme yapabiliyor olmasi

cv_results['test_accuracy'].mean()
# 0.733112638994992
cv_results['test_f1'].mean()
# 0.5905780011534191
cv_results['test_roc_auc'].mean()
# 0.7805279524807827

# Basari skorlari nasil arttirilabilir
# 1. Ornek boyutu arttirilabilir.
# 2. Veri on islme
# 3. Ozellik muhendisligi
# 4. Ilgili algoritma icin optimizasyonlar yapilabilir.

knn_model.get_params()

# 5. Hyperparameter Optimization

knn_model = KNeighborsClassifier()
knn_model.get_params()

#
knn_params = {'n_neighbors': range(2,50)}

# GridSearchCv en optimal secenekleri arar. Burada cv modeli kurup hatasina bakma islemini de cv kadar yapmasini
# istiyoruz. n_jobs=-1 islemcileri tam performans kullanir. Verbose rapor bekliyor musun diye sorar.
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
# Fitting 5 folds for each of 48 candidates, totalling 240 fits.

# optimal deger
knn_gs_best.best_params_

# 6. Final Model

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7669892199303965
cv_results['test_f1'].mean()
# 0.6170909049720137
cv_results['test_roc_auc'].mean()
# 0.8127938504542278

random_user = X.sample()

knn_final.predict(random_user)