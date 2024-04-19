import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import norm, boxcox
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("Miull/Projects_doc/winequalityN.csv")


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


def outlier_thresholds(dataframe, col_name, q1, q3):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable, q1, q3):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col, 0.05, 0.95)

df["type"] = LabelEncoder().fit_transform(df["type"])

for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)

# Bu kod, verileri yeniden ölçeklendirmek için Box-Cox transformasyonunu uygulamak için kullanılır.
# [1] Box-Cox transformasyonu, bir veri kümesinin dağılımının normalleştirilmesi için kullanılır.
# [2] Normalleşme, bir veri kümesinin aritmetik ortalamasının 0 ve varyansının 1 olmasıyla elde edilir.
# [3] [1] https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing.scale
# [2] https://en.wikipedia.org/wiki/Normal_distribution [3] https://www.statisticshowto.com/normalization/

df["total acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]

df["residul sugar levels"] = pd.cut(df["residual sugar"],
                                    bins=[0, 1, 17, 35, 120, 1000],
                                    labels=["bone_dry", "dry", "off_dry", "medium_dry", "sweet"])

df["alcohol levels"] = pd.cut(df["alcohol"], bins=[0, 12.5, 13.5, 14.5, 20],
                              labels=["low", "moderately_low", "high", "very_high"])

df["white perfect pH"] = np.where((df["type"] == 1) & (df["pH"] >= 3) & (df["pH"] <= 3.3), 1, 0)

df["red perfect pH"] = np.where((df["type"] == 0) & (df["pH"] >= 3.3) & (df["pH"] <= 3.6), 1, 0)

df["fixed acidity ratio"] = df["fixed acidity"] / df["total acidity"]

df["volatile acidity ratio"] = df["volatile acidity"] / df["total acidity"]

df["free sulfur ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]

df["quality"] = pd.cut(df["quality"], bins=[0, 5, 6, 10], labels=["bad", "good", "perfect"])

df["fixed acidity"], lam = boxcox(df["fixed acidity"])

df["residual sugar"], lam_fixed_acidity = boxcox(df["residual sugar"])

df["free sulfur dioxide"], lam_fixed_acidity = boxcox(df["free sulfur dioxide"])

df["total sulfur dioxide"], lam_fixed_acidity = boxcox(df["total sulfur dioxide"])

df["alcohol"], lam_fixed_acidity = boxcox(df["alcohol"])



df["residul sugar levels"] = LabelEncoder().fit_transform(df["residul sugar levels"])
df["alcohol levels"] = LabelEncoder().fit_transform(df["alcohol levels"])
df["quality"] = LabelEncoder().fit_transform(df["quality"])



y = df["quality"]
X = df.drop('quality',axis=1)

sc = StandardScaler()
X = sc.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.grid()
plt.show()

pca_new = PCA(n_components=11)
X_new = pca_new.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20)

sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy : ', test_data_accuracy)

print("Classification Report:")
print(classification_report(y_test, X_test_prediction))
