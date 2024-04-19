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
import catboost as cb
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, log_loss
from scipy.stats import norm, boxcox

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

df["type"] = LabelEncoder().fit_transform(df["type"])

# ----------------------------------------------------------------------------------------------------------------------

RF = OneVsRestClassifier(RandomForestClassifier())
RF.fit(X_train,y_train)
y_pred = RF.predict(X_test)
pred_prob=RF.predict_proba(X_test)

RF.score(X_test,y_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------------------------------------------------------


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

df.isnull().sum()

for col in df.columns:
    df[col].fillna(df[col].median(), inplace=True)


df["fixed acidity"], lam = boxcox(df["fixed acidity"])

df["residual sugar"], lam_fixed_acidity = boxcox(df["residual sugar"])

df["free sulfur dioxide"], lam_fixed_acidity = boxcox(df["free sulfur dioxide"])

df["total sulfur dioxide"], lam_fixed_acidity = boxcox(df["total sulfur dioxide"])

df["alcohol"], lam_fixed_acidity = boxcox(df["alcohol"])

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

df["perfect ph"] = df["white perfect pH"] + df["red perfect pH"]

df["Cquality"] = pd.cut(df["quality"], bins=[0, 5, 6, 10], labels=["bad", "good", "perfect"])


df["residul sugar levels"] = LabelEncoder().fit_transform(df["residul sugar levels"])
df["alcohol levels"] = LabelEncoder().fit_transform(df["alcohol levels"])
df["Cquality"] = LabelEncoder().fit_transform(df["Cquality"])

df.drop(columns=["quality"], inplace=True, axis=1)

y = df["Cquality"]
X = df.drop(["Cquality"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)


model = RandomForestClassifier()
model.fit(X_train, y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy : ', test_data_accuracy)

print("Classification Report:")
print(classification_report(y_test, X_test_prediction))


# Initialize Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, min_samples_split=5)

# Fit the classifier to the training data
gb_classifier.fit(X_train, y_train)

# Predict on the testing data
y_pred = gb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# oversample = RandomOverSampler(sampling_strategy='minority')
# X_randomover, y_randomover = oversample.fit_resample(X_train, y_train)
# gb_classifier.fit(X_randomover, y_randomover)

# oversample = SMOTE()
# X_smote, y_smote = oversample.fit_resample(X_train, y_train)
# gb_classifier.fit(X_smote, y_smote)

parameters_KNN = {
    "n_neighbors": [2, 5, 7, 15],
    "weights": ('uniform', 'distance'),
    "algorithm": ('auto', 'ball_tree', 'kd_tree', 'brute'),
    'p': [1, 2, 5]}

model_KNN = KNeighborsClassifier (n_jobs=-1)
model_KNN_with_best_params = GridSearchCV(model_KNN, parameters_KNN)
model_KNN_with_best_params.fit(X_train, y_train)
model_KNN_best_params = model_KNN_with_best_params.best_params_

model_KNN_best_params

predictions_KNN = model_KNN_with_best_params.predict(X_test)
print("Predictions:",predictions_KNN[:10])
print("Actual:",y_test[:10])

# {'algorithm': 'auto', 'n_neighbors': 15, 'p': 1, 'weights': 'distance'}
accuracy = accuracy_score(y_test, predictions_KNN)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, predictions_KNN))

# -----------

#XGB model building
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model.fit(X_train, y_train)

#model
#print('Fit time : ', time.time() - start_time)

import optuna
from sklearn.metrics import mean_squared_error, log_loss


# step 1
def objective(trial):
    # a. defining possible parameters in a dictionary
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 5),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
        # 'multi_class': 'multinomial',  # Specify 'multinomial' for multi-class classification
        # 'num_class': 6
    }

    # b. intializing and fitting the model
    optuna_model = xgb.XGBClassifier(**params)  # XGBClassifier(**params)
    optuna_model.fit(X_train, y_train)

    # c. predicting the results using X_val and generating the stats
    # prediction_val = optuna_model.predict(X_val)

    prediction_val_proba = optuna_model.predict_proba(X_val)
    # rmse_score = mean_squared_error(y_val, prediction_val, squared=False)  # Calculate RMSE

    score = log_loss(y_val, prediction_val_proba, labels=optuna_model.classes_)
    return score

study = optuna.create_study(direction='minimize')

#step3: optimizing study
study.optimize(objective, n_trials=30)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

trial = study.best_trial

print('Number of finished trials:', len(study.trials))
print('Best trial:')
print('Value: {}'.format(trial.value))
print('Params: ')
for key, value in trial.params.items():
    print('{}: {}'.format(key, value))

best = study.best_params
final_model = xgb.XGBClassifier(best)
final_model.fit(X_train, y_train)
result = final_model.predict(X_test)

le = LabelEncoder()
y = le.fit_transform(y)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

model.fit(X_train, y_train)

import optuna
from sklearn.metrics import mean_squared_error, log_loss


# step 1
def objective(trial):
    # a. defining possible parameters in a dictionary
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.04, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 5),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0),
        # 'multi_class': 'multinomial',  # Specify 'multinomial' for multi-class classification
        # 'num_class': 6
    }

    # b. intializing and fitting the model
    optuna_model = xgb.XGBClassifier(**params)  # XGBClassifier(**params)
    optuna_model.fit(X_train, y_train)

    # c. predicting the results using X_val and generating the stats
    # prediction_val = optuna_model.predict(X_val)

    prediction_test_proba = optuna_model.predict_proba(X_test)
    # rmse_score = mean_squared_error(y_val, prediction_val, squared=False)  # Calculate RMSE

    score = log_loss(y_test, prediction_test_proba, labels=optuna_model.classes_)
    return score

study = optuna.create_study(direction='minimize')

#step3: optimizing study
study.optimize(objective, n_trials=30)

trial = study.best_trial

print('Number of finished trials:', len(study.trials))
print('Best trial:')
print('Value: {}'.format(trial.value))
print('Params: ')
for key, value in trial.params.items():
    print('{}: {}'.format(key, value))

best = study.best_params
final_model = xgb.XGBClassifier(best)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

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


plot_importance(rf_model, X_train)