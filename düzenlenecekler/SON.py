import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay,recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
import warnings
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.stats import norm, boxcox
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from scipy.stats import skew, norm, probplot, boxcox, f_oneway
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, accuracy_score

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


df.dropna(inplace=True)

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

df["total acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]

df["alcohol levels"] = pd.cut(df["alcohol"], bins=[0, 12.5, 13.5, 14.5, 20],
                              labels=["low", "moderately_low", "high", "very_high"])


df["fixed acidity ratio"] = df["fixed acidity"] / df["total acidity"]

df["volatile acidity ratio"] = df["volatile acidity"] / df["total acidity"]

df["free sulfur ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]

df["quality"] = pd.cut(df["quality"], bins=[0, 5, 6, 10], labels=["bad", "good", "perfect"])

df["fixed acidity"], lam = boxcox(df["fixed acidity"])

df["residual sugar"], lam_fixed_acidity = boxcox(df["residual sugar"])

df["free sulfur dioxide"], lam_fixed_acidity = boxcox(df["free sulfur dioxide"])

df["total sulfur dioxide"], lam_fixed_acidity = boxcox(df["total sulfur dioxide"])

df["alcohol"], lam_fixed_acidity = boxcox(df["alcohol"])

df["alcohol levels"] = LabelEncoder().fit_transform(df["alcohol levels"])
df["quality"] = LabelEncoder().fit_transform(df["quality"])

df.drop("type",axis=1, inplace=True)

y = df["quality"]
X = df.drop(["quality"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,
                               cv=3,
                               verbose=True,
                               random_state=17,
                               n_jobs=-1)


rf_random.fit(X, y)

rf_random.best_params_

#Out[33]:
#best_params = {'n_estimators': 344, 'min_samples_split': 22,'max_features': 3,'max_depth': 49}

rf_random_final = rf_model.set_params(n_estimators= 344, min_samples_split= 22,max_features= 3,max_depth= 49).fit(X, y)

X_test_prediction = rf_random_final.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy : ', test_data_accuracy)

print("Classification Report:")
print(classification_report(y_test, X_test_prediction))
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_random_final, X_train)


rf_random_final = rf_model.set_params(n_estimators= 633, min_samples_split= 43,max_features= "auto",max_depth=33).fit(X, y)
rf_random_final = rf_model.set_params(n_estimators= 200, min_samples_split= 38,max_features= 7,max_depth=36,random_state=17).fit(X, y)

X_test_prediction = rf_random_final.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy : ', test_data_accuracy)

print("Classification Report:")
print(classification_report(y_test, X_test_prediction))

scoring = {'accuracy_scorer' : make_scorer(accuracy_score)}

results = cross_validate(estimator = rf_random_final,
                          X = X_train,
                          y = y_train,
                          cv = 3,
                          scoring = scoring)

results['test_accuracy_scorer'].mean()

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_random_final, X_train)