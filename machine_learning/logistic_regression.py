# Logistic Regression

# Bir siniflandirma problemi icin bagimli ve bagimsiz degiskenler arasindaki iliskiyi dogrusal olarak modellemektir.
# Optimizasyonu gercek degerler ile tahmin edilen degerler arasindaki farklara iliskin log loss degerini minimum
# yapabilecek agirliklari bularak. Siniflandirma problemlerinde buldumuz sonuc 1 degeri olma olasiligi olarak bulunur.

# Siniflandirma Projelerinde Basari Degerlendirme

# Confusion Matrix
# Accuracy: Dogru Siniflandirma Oranidir
# Precision: Pozitif tahminlerin basari oranidir.
# Recall: Pozitif sinifin dogru tahmin edilme oranidir.
# Sinifler dengesiz dagiliyorsa Accuracy skorunu kullanamayiz. Recall ve Precision degerine bakmamiz gerekir.

# Classification Threshold : esik degeri olarak dusunebiliriz. Yukari cekersek accuracynin dusme ihtimali yuksektir.

# ROC Curve (Receiver Operating Characteristic Curve)(
# Area Under Curve (AUC) : ROC egrisinin tek bir sayisal deger ile ifade edilisidir. ROC egrisinin altinda kalan
# alandir. AUC tum olasi siniflandirma esikleri icin toplu bir performans olcusudur.

# Eger veri seti dengesizse (5-95 gibi) bu durumda recall precision ve f1 degerine bakiyoruz. AUC de bakariz.

# LOG Loss: Bir basari metrigidir. Optimizasyon yapmak uzere agirliklari bulurken optimize etmek icin kullanacagimiz
# fonksiyondur.
# Entropi cesitliliktir. Bilgidir.







# Diabetes Prediction with Logistic Regression


# Is Problemi:

# Ozellikleri belirtildiginde kisilerin diyabet hastasi olup
# olmadiklarini tahmin edebilecek bir makine ogrenmesi
# modeli gelistirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Bobrek Hastalıkları Enstituleri'nde tutulan buyuk veri setinin
# parcasidir. ABD'deki Arizona Eyaleti'nin en buyuk 5. sehri olan Phoenix sehrinde yasayan 21 yas ve uzerinde olan
# Pima Indian kadinlari uzerinde yapilan diyabet arastirmasi icin kullanilan verilerdir. 768 gozlem ve 8 sayisal
# bagimsiz degiskenden olusmaktadir. Hedef degisken "outcome" olarak belirtilmis olup; 1 diyabet test sonucunun
# pozitif olusunu, 0 ise negatif olusunu belirtmektedir.

# Degiskenler
# Pregnancies: Hamilelik sayisi
# Glucose: Glikoz.
# BloodPressure: Kan basınci.
# SkinThickness: Cilt Kalınligi
# Insulin: Insulin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kisilere gore diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yas (yil)
# Outcome: Kisinin diyabet olup olmadigi bilgisi. Hastaliga sahip (1) ya da degil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# 1. Exploratory Data Analysis

df = pd.read_csv("Miull/Feature Engineering/diabetes/diabetes.csv")

# Analyses of the Target

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

# Analyses of theFeatures

df.head()

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)
# Grafiklerin ust uste binmemesi icin Block=True kullaniriz.

for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]


# for col in cols:
#     plot_numerical_col(df, col)

df.describe().T

# Target vs Features

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)

# Data Preprocessing (Veri On Isleme)

df.isnull().sum()
df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

# Robust aykiri degerlere daha dayanikli. Ortalamayi degil de medyani cikariyor.

df.head()

# Model & Prediction

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X,y)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

# Model Evaluation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy = 0.78
# Precision = 0.74
# Recall = 0.58
# F1-score = 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.8394
# Ayni set uzerinden hem modeli kurup hem de test ettigimiz icin kontrol etmemiz gerekir.

# Model Validation: Holdout

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]    # Bir sinifin ait olma olasiligi

print(classification_report(y_test, y_pred))

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
# Hangi 80-20 oldugunu bilmedigimizden sonuclari dogrulamak icin 10 katli capraz dogrulama yapacagiz.

# Model Validation: 10-Fold Cross Validation

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X,y)

# Veri setimiz bol ise veri setini bastan hold outtaki gibi train test diye ikiye ayirip 10 katli validasyon yapip en
# son bir de test seti performansi incelenebilir. Eger bol degilse butun veriyi kullanarak da bu capraz dogrulama
# islemini gerceklestirebiliriz.

cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()

cv_results["test_precision"].mean()

cv_results["test_recall"].mean()

cv_results["test_f1"].mean()

cv_results["test_roc_auc"].mean()

