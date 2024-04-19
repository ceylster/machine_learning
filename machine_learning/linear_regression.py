# Linear Regression (Dogrusal Regresyon)

# Amac, bagimli ve bagimsiz degiskenler arasindaki iliskiyi dogrusal olarak modellemektedir.
# Gercek degerler ile tahmin edilen degerler arasindaki farklarin karelerinin toplamini/ortalamasini minimum yapabilecek
# b ve w degerlerini bularak agirlik degerlerini buluruz.

# Regresyon Modellerinde Basari Degerlendirme

# - MSE (Hata kareler ortalamasi (Ilgilenilen problemin bagimli degiskeninin ortalamasina yakin bir yerde olmasi kabul
# edilebilir olur.)
# - RMSE (Hata kareler ortalamasi karekoku)
# - MAE (Mutlak ortalama hata) (Mean Absolute Error)

# Parametrelerin Tahmin Edilmesi (Agirliklarin Bulunmasi, Parametre Tahmincilerinin Bulunmasi)

# Analitik Cozum: Normal Denklemler Yontemi(En Kucuk Kareler Yontemi)
# Optimizasyon Yontemi: Gradient Descent (Optimizasyon yontemidir. Bir fonksiyonu minimum yapacak parametreleri
# optimize etmeyi saglar.)
# Gradyanin yani turevin negatif olarak tanimlanan en dik inis yonunde iteratif olarak parametre degerlerini
# guncelleyerek ilgili fonksiyonun minimum degerini verebilecek parametreleri bulur.

# Sales Prediction with Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Simple Linear Regression with OLS Using Scikit-Learn

df = pd.read_csv("Miull/Projects_doc/linearregresiondatasets/advertising.csv")

X = df[["TV"]]
y = df[["sales"]]

# Model

reg_model = LinearRegression().fit(X,y)

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayisi (w1)
reg_model.coef_[0][0]

# Tahmin

# 150 birimlik TV harcamasi olsa ne kadar satis olmasi beklenir.

reg_model.intercept_[0] + reg_model.coef_[0][0]*150

# 500 birimlik TV harcamasi olsa ne kadar satis olur?

reg_model.intercept_[0] + reg_model.coef_[0][0]*500

df.describe().T

# Modelin Gorsellestirilmesi

g = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s": 9}, ci=False, color="r")    # ci guven araligini ifade eder.

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satis Sayisi")
g.set_xlabel("TV Harcamalari")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

# Tahmin Basarisi

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-kare (Veri setindeki bagimsiz degiskenlerin bagimli degiskeni aciklama yuzdesidir.)
reg_model.score(X, y)

# Multiple Linear Regression

X = df.drop("sales", axis=1)

y = df[["sales"]]

# Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# reg_model = LinearRegression()
# reg_model.fit(X_train, y_train)
reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b-bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

# Tahmin

# Asagidaki gozlem degerlerine gore satisin beklenen degeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

reg_model.intercept_ + 30 * reg_model.coef_[0][0] + 10 * reg_model.coef_[0][1] + 40 * reg_model.coef_[0][2]

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)

# Tahmin Basarisi Degerlendirme

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Train RKARE
reg_model.score(X_train, y_train)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Normalde test hatasi train hatasindan daha yuksek cikar.

# Test RKARE
reg_model.score(X_test, y_test)

# 10 Katli CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))  # neg oldugu icin - ile.

# 5 Katli CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# veri seti az oldugundan 10 yerine 5 yapsak daha mi dogru olur.

# Simple Linear Regression with Gradient Descent from Scratch

# Cost Function
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_devriv_sum = 0
    w_devriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_devriv_sum += (y_hat - y)
        w_devriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_devriv_sum)
    new_w = w - (learning_rate * 1 / m * w_devriv_sum)
    return new_b, new_w

# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse{2}".format(initial_b, initial_w,
                                                                         cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter={:d}  b={:.2f}  w={:.4f}  mse={:.4}".format(i, b, w, mse))
    print("After {0} iterations b = {1}, w={2}, mse={3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

df = pd.read_csv("Miull/Projects_doc/linearregresiondatasets/advertising.csv")

X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)