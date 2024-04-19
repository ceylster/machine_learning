# Principal Component Analysis (Temel Bilesen Analizi)

# Temel fikir, cok degiskenli verinin ana ozelliklerini daha az sayida degisken/bilesen ile temsil etmektir. Diger bir
# ifadeyle kucuk miktarda bir bilgi kaybini goze alip degisken boyutunu azaltmaktir.
# Temel bilesen analizi veri setinin bagimzsiz degiskenlerin dogrusal kombinasyonlari ile ifade edilen bilesenlere
# indirger. Dolayisiyla bu bilesenler arasinda korelasyon yoktur. Bu bilesenler korelasyonsuzdur. Varyansa, oz degerlere
# dayali bir gruplama yapilmaktadir.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV


df = pd.read_csv("Miull/Projects_doc/hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df = df[num_cols]
df.dropna(inplace=True)

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

# Bilensenlerin basarisi acikladiklari varyans oranina gore belirlenmektedir.

pca.explained_variance_ratio_

# Pespese iki bilesenin aciklayacak oldugu varyans nedir?

np.cumsum(pca.explained_variance_ratio_)

# Optimum Bilesen sayisi

# Bu yontemle en keskin gecisin nerede oldugunu inceleyebiliyorduk. Veri gorsellestirmek istiyorsak 2 boyuta indirmek
# zorundayiz.
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bilesen Sayisi")
plt.ylabel("Kumulatif Varyans Orani")
plt.show()

# 3 sayisini secmis olalim.

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

# Principal Component Regression (Temel Regresyon Yontemi)

# Once bir temel bilesen yontemi uygulanip degiskenlerin boyutu indirgeniyor. Daha sonrasinda bu bilesenlerin uzerine
# bir regresyon modeli kuruluyor.

# Dogrusal bir modelde degiskenler arasinda coklu dogrusal baglanti problemi oldugunu dusunelim. Degiskenler arasinda
# yuksek korelasyon oldugunda bu cesitli problemlere sebep olur. Bunu istemiyoruz.

df = pd.read_csv("Miull/Projects_doc/hitters.csv")
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]), df[others]], axis=1)   # axis=1 yanyana

def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "League", "Division"]:
    label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()

rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))

y.mean()

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {"max_depth": range(1, 11),
               "min_samples_leaf": range(2, 20)}

# GridSearchCV

cart_best_grid = GridSearchCV(cart, cart_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X,y)
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))

# PCA ile Cok Boyutlu Veriyi 2 Boyutta Gorsellestirme

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

df = pd.read_csv("Miull/Projects_doc/breast_cancer.csv")

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)  # Bagimsiz degiskenleri standartlastiracak.
    pca = PCA(n_components=2)    # Daha sonra PCA hesabi yapcak
    pca_fit = pca.fit_transform(X)   # Degisken degerlerini donusturecek
    pca_df = pd.DataFrame(data=pca_fit, columns=["PC1", "PC2"])   # olusan bilesenleri DataFrame'e cevirecek.
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)  # ve bagimli degisken ile concat ederek disari r.
    return final_df

pca_df = create_pca_df(X, y)

# bu iki bileseni gorsellestirmemiz gerekiyor. Alttaki fonksiyonu kullanacagiz
def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")

###############

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")

############

df = pd.read_csv("Miull/Projects_doc/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")
