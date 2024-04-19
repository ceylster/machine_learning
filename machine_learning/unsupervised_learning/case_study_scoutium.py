import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
# Adim 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarini okutunuz.

df1 = pd.read_csv("Miull/Projects_doc/scoutium_attributes.csv", delimiter=";")
df2 = pd.read_csv("Miull/Projects_doc/scoutium_potential_labels.csv", delimiter=";")

# Adim 2: Okutmus oldugumuz csv dosyalarini merge fonksiyonunu kullanarak birlestiriniz. ("task_response_id",
# 'match_id', 'evaluator_id' "player_id" 4 adet degisken uzerinden birlestirme islemini gerceklestiriniz.)

df = pd.merge(df1, df2, on=["task_response_id", "match_id", "evaluator_id", "player_id"])

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

# Adim 3: position_id icerisindeki Kaleci (1) sinifini veri setinden kaldiriniz.

df.drop(df[df["position_id"] == 1].index, inplace=True)

#  Adim 4: potential_label icerisindeki below_average sinifini veri setinden kaldiriniz.(below_average sinifi tum
#  verisetinin %1'ini olusturur)

df.drop(df[df["potential_label"] == "below_average"].index, inplace=True)

# Adim 5: Olusturdugunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo olusturunuz. Bu pivot table'da
# her satirda bir oyuncu olacak sekilde manipulasyon yapiniz.
#   Adim 1: Indekste “player_id”,“position_id” ve “potential_label”, sutunlarda “attribute_id” ve degerlerde scout’larin
#   oyunculara verdigi puan “attribute_value” olacak sekilde pivot table’i olusturunuz.

pivotdf = df.pivot_table(index=["player_id", "position_id", "potential_label"],
                         columns="attribute_id",
                         values="attribute_value")

# Adim 2: “reset_index” fonksiyonunu kullanarak indeksleri degisken olarak atayiniz ve “attribute_id” sutunlarinin
# isimlerini stringe ceviriniz.

pivotdf.columns = pivotdf.columns.astype(str)

pivotdf.reset_index(inplace=True)

# Adim 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayisal olarak
# ifade ediniz.

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

label_encoder(pivotdf, "potential_label")


# Adim 7: Sayisal degisken kolonlarini “num_cols” adiyla bir listeye atayiniz.

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

cat_cols, num_cols, cat_but_car = grab_col_names(pivotdf)

num_cols = [col for col in num_cols if col not in "player_id"]

# Adim 8: Kaydettiginiz butun “num_cols” degiskenlerindeki veriyi olceklendirmek icin StandardScaler uygulayiniz.

scaled = StandardScaler()
pivotdf[num_cols] = scaled.fit_transform(pivotdf[num_cols])

# Adim 9: Elimizdeki veri seti uzerinden minimum hata ile futbolcularin potansiyel etiketlerini tahmin eden bir makine
# ogrenmesi modeli gelistiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdiriniz.)
y = pivotdf["potential_label"]
X = pivotdf[num_cols]

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   #('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   ('CatBoost', CatBoostClassifier(verbose=False))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# roc_auc için tüm model skorlarını görelim
base_models(X, y, scoring="roc_auc")

# f1 skor için model kuralım
base_models(X, y, scoring="f1")

# precision skor için model kuralım
base_models(X, y, scoring="precision")

# recall skor için model kuralım
base_models(X, y, scoring="recall")

# accuracy için tüm model skorlarını görelim
base_models(X, y, scoring="accuracy")

# RF, LightGBM ve Catboost en iyi modeller gibi gozukuyor.


rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

classifiers = [('CatBoost', CatBoostClassifier(verbose=False), catboost_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=10, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y, scoring="roc_auc")

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('CatBoost', best_models["CatBoost"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                               voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_classifier(best_models, X, y)

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model= RandomForestClassifier()
model.fit(X, y)

plot_importance(model, X)