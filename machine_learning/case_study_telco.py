import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Telco Churn Prediction

# Is Problemi
# Sirketi terk edecek musterileri tahmin edebilecek bir makine ogrenmesi modeli gelistirilmesi beklenmektedir.

# Veri Seti Hikayesi
# Telco musteri kaybi verileri, ucuncu ceyrekte Kaliforniya'daki 7043 musteriye ev telefonu ve Internet hizmetleri
# saglayan hayali bir telekom sirketi hakkinda bilgi icerir. Hangi musterilerin hizmetlerinden ayrildigini,
# kaldigini veya hizmete kaydoldugunu gosterir.

df = pd.read_csv("Miull/Projects_doc/Telco-Customer-Churn.csv")

# df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df = df[~(df["TotalCharges"] == " ")]
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
def check_df(dataframe, head=5):
    print(10*"#" + " Shape ".center(9) + 10*"#")
    print(dataframe.shape)
    print(10*"#" + " Types ".center(9) + 10*"#")
    print(dataframe.dtypes)
    print(10*"#" + " Head ".center(9) + 10*"#")
    print(dataframe.head(head))
    print(10*"#" + " Tail ".center(9) + 10*"#")
    print(dataframe.tail(head))
    print(10*"#" + " NA ".center(9) + 10*"#")
    print(dataframe.isnull().sum())
    print(10*"#" + " Quantiles ".center(9) + 10*"#")
    print(dataframe.describe([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]).T)
    print(10*"#" + " Unique Values ".center(9) + 10*"#")
    print(dataframe.nunique())
check_df(df)

# Adim 1: Numerik ve kategorik degiskenleri yakalayiniz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe:dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th:int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal degişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişen listesi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un icerisinde.


    """


    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["object", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64", "int32", "float32"]]
    num_cols = [col for col in dataframe.columns if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_car: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


num_cols = [col for col in num_cols if col not in "customerID"]

# Adim 2: Gerekli duzenlemeleri yapiniz. (Tip hatasi olan degiskenler gibi)
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

# Adim 3: Numerik ve kategorik degiskenlerin veri icindeki dagilimini gozlemleyiniz.

def cat_summary(dataframe,col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

# Adim 4: Kategorik degiskenler ile hedef degisken incelemesini yapiniz.


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

# Adim 5: Aykiri gözlem var mi inceleyiniz.

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

for col in num_cols:
    print(col, check_outlier(df,col))

for col in num_cols:
    col, outlier_thresholds(df,col)

# Adim 6: Eksik gozlem var mi inceleyiniz.
df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

# df[df["TotalCharges"] == " "]
# df["TotalCharges"] = pd.to_numeric(Df["TotalCharges", errors="coerce")
#df = df[~(df["TotalCharges"] == " ")]
#df["TotalCharges"] = df["TotalCharges"].astype(float)

# Gorev 2 : Feature Engineering

# Adim 1:  Eksik ve aykiri gozlemler icin gerekli islemleri yapiniz.
# df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Adim 2: Yeni degiskenler olusturunuz.

df[num_cols].corr()
sns.set(rc={'figure.figsize': (18,13)})
sns.heatmap(df[num_cols].corr(), cmap="RdBu", annot=True, fmt=".2f")
plt.show(block=True)

df["AverageMonthlyBill"] = df["TotalCharges"] / df["tenure"]

df["Tenure_Year"] = pd.qcut(df["tenure"], q=6, labels=["0_1_Year", "1_2_Year", "2_3_Year", "3_4_Year", "4_5_Year", "5_6_Year"])

df["New_noProt"] = df.apply(lambda x:1 if (x["OnlineBackup"] != "Yes") or (x["TechSupport"] != "Yes") or
                                          (x["DeviceProtection"] != "Yes") else 0, axis=1)

# axis=1 ifadesi, apply() fonksiyonunun her satır için uygulanacağını belirtir.

df["New_Flag_Any_Streaming"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or
                                                       (x["StreamingMovies"] == "Yes") else 0, axis=1)

df["New_Total_Servives"] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                               'OnlineBackup', 'DeviceProtection', 'TechSupport',
                               'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

# Adim 3: Encoding islemlerini gerceklestiriniz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True, dtype=int):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


one_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]

df = one_hot_encoder(df, one_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "customerID"]

mms = MinMaxScaler()
df[num_cols] = mms.fit_transform(df[num_cols])


# Gorev 3 : Modelleme
y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

log_model = LogisticRegression().fit(X,y)

log_model = LogisticRegression().fit(X,y)
log_model.intercept_
log_model.coef_
y_pred = log_model.predict(X)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()

cv_results["test_precision"].mean()

cv_results["test_recall"].mean()

cv_results["test_f1"].mean()

cv_results["test_roc_auc"].mean()