# Recommendation Systems (Tavsiye Sistemleri)

# Kullanicilara bazi teknikleri kullanarak urun ya da hizmet onermek ya da tavsiye etme sistemidir.

# Simple Recommender Systems (Basit Tavsiye Sistemleri)
#  - Is bilgisi ya da basit tekniklerle yapilan genel oneriler.
#  - Kategorinin en yuksek puanlilari, trend olanlar, efsaneler vs.

# Association Rule Learning (Birliktelik Kurali Ogrenimi)
#  - Birliktelik analizi ile ogrenilen kurallara gore urun onerileri.

# Content Based Filtering (Icerik Temelli Filtreleme)
#  - Urun benzerligine gore oneriler yapilan uzaklik temelli yontemler.

# Collaborative Filtering (Is Birlikci Filtreleme)
#  - Toplulugun kullanici ya da urun bazinda ortak kanaatlerini yansitan yontemler.
#  - User-Based (Memory-Based, Neighborhood Methods)
#  - Item-Based (Memory-Based, Neighborhood Methods)
#  - Model-Based (Matrix Factorization)


# Association Rule Learning (Birliktelik Kurali Ogrenimi)

# Veri icersindeki oruntuleri(pattern, iliski, yapi) bulmak icin kullanilan kural tabanli bir makine ogrenmesi
# teknigidir.

# Apriori Algoritmasi (Apriori Algorithm)

# Sepet analizi yontemidir. Urun birlikteliklerini ortaya cikarmak icin kullanilir.
# Support(X, Y) = Freq(X, Y)/N (X ve Y'nin birlikte gorulme olasiligi)
# Connfidence(X, Y) = Freq(X, Y) / Freg(X) (X satin alindiginda Y'nin satilmasi olasiligi)
# Lift = Support(X, Y) / (Support(X) * Support(Y)) (X satin alindiginda Y'nin satin alinma olasiligi lif kadar ARTAR.)


# Association Rule Based Recommender System (Birliktelik Kurali Temelli Tavsiye Sistemi)
# Association Rule Learning (Birliktelik Kurali Ogrenimi)

# 1. Veri On Isleme
# 2. ARL Veri Yapisini Hazirlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarinin Cikarilmasi
# 4. Calismanin Scriptini Hazirlama
# 5. Sepet Asamasindaki Kullanicilara Urun Onerisinde Bulunmak


# 1. Veri On Isleme
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)  # Ciktinin tek bir satirda olmasini saglar.
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("Miull/Projects_doc/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

# pip install openpyxl
# df_ = pd.read_excel("Miull/Projects_doc/online_retail_II.xlsx", sheet_name="Year 2010-2011", engine="openpyxl")

df.head()
df.isnull().sum()


def retail_date_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe


df = retail_date_prep(df)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_date_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_date_prep(df)
df.isnull().sum()
df.describe().T

# 2. Preparing ARL Data Structures (ARL Veri Yapilarini Hazirlamak)

# Invoice-Product Matrix yapmak istiyoruz

df_fr = df[df["Country"] == "France"]

df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).head(20)


# df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
# unstack islemi pivota cevirmeye yarar.

# df_fr.groupby(["Invoice", "Description"]).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# df_fr.groupby(["Invoice", "Description"]). \
#           agg({"Quantity": "sum"}).unstack(). \
#           fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# df_fr.groupby(["Invoice", "StockCode"]). \
#     agg({"Quantity": "sum"}).unstack(). \
#     fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    produckt_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(produckt_name)


check_id(df_fr, 11001)

# 3. Association Rules Analysis (Birliktelik Kurallari Analizi)

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemset, metric="support", min_threshold=0.01)
# leverage=kaldirac etkisidir.
# conviction=x urunu olmadan y urununun beklenen performansidir.

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.01) & (rules["lift"] > 5)]

check_id(df_fr, 21086)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.01) & (rules["lift"] > 5)]. \
    sort_values("confidence", ascending=False)


# 4. Preparin the Script of the Study (Calismanin Scriptini Hazirlama)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

# 5. Product Recommendation Practice (Urun Onerme Uygulamasi)

# Ornek:
# Kullanici ornek urun id: 22492

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0]

check_id(df, 22556)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]
arl_recommender(rules, 22492,1)
arl_recommender(rules, 22492,2)

