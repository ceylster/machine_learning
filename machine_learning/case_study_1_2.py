import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Regresyon Modelleri icin Hata Degerlendirme

df = pd.read_excel("Miull/Projects_doc/machine_learning_case1_case2.xlsx", sheet_name="CaseStudy1")

# Adim 1:Verilen bias ve weight’e gore dogrusal regresyon model denklemini olusturunuz. Bias=275, Weight=90(y’=b+wx)

X = df[["Deneyim Yili"]]
y = df[["Maas"]]

y = 270 + 90*x

# Adim 2: Olusturdugunuz model denklemine gore tablodaki tum deneyim yillari icin maas tahmini yapiniz

# numbers = np.arange(1,11)
# z = 270 + 90*numbers

y_pred = 270 + 90*X

# Adim 3: Modelin basarisiin olcmek icin MSE, RMSE, MAE skorlarini hesaplayiniz.

# MSE
mean_squared_error(y, y_pred)

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# Siniflandirma Modeli Degerlendirme

df = pd.read_excel("Miull/Projects_doc/machine_learning_case1_case2.xlsx", sheet_name="CaseStudy2")

df.drop("Unnamed: 0", axis=1, inplace=True)

# Esik degerini 0.5 alarak confusion matrix olusturunuz.

y = df[["Gercek Deger"]]
y_pred = df[["Model Olasilik Tahmini"]]
# 5   1
# 1   3
# accuracy=0.8
# precision = 0.83
# recall = 0.83
# f1 = 0.83
