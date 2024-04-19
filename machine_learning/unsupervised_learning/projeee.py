import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
import matplotlib

warnings.simplefilter(action="ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


# Adim 1: flo_data_20K.csv verisini okutunuz.

df = pd.read_csv("Miull/Projects_doc/flo_data_20k.csv")

# Adim 2: Musterileri segmentlerken kullanacaginiz degiskenleri seciniz.

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df["last_order_date"].max()

analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).dt.days

df["tenure"] = (df["last_order_date"]-df["first_order_date"]).dt.days

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online", "recency", "tenure"]]

# Gorev 2: K-Means ile musteri Segmentasyonu

# Adim 1: Degiskenleri standartlastiriniz.
#SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)

# Normal dagilimin saglanmasi icin Log transformation uygulanmasi

model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()

# Scaling

sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()

# Adim 2: Optimum kume sayisini belirleyiniz.

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)

# Adim 3: Modelinizi olusturunuz ve musterilerinizi segmentleyiniz.

k_means = KMeans(n_clusters=7).fit(model_df)
segments = k_means.labels_
segments

final_df = df[["master_id", "order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online", "recency", "tenure"]]
final_df["segment"] = segments
final_df.head()

# Adim 4: Herbir segmenti istatistiksel olarak inceleyeniz.

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                   "order_num_total_ever_offline":["mean","min","max"],
                                   "customer_value_total_ever_offline":["mean","min","max"],
                                   "customer_value_total_ever_online":["mean","min","max"],
                                   "recency":["mean","min","max"],
                                   "tenure":["mean","min","max","count"]})

# Gorev 3: Hierarchical Clustering ile Musteri Segmentasyonu

# Adim 1: Gorev 2'de standırlastirdiginiz dataframe'i kullanarak optimum kume sayisini belirleyiniz.

model_df.info()
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)

# Adim 2: Modelinizi olusturunuz ve musterileriniz segmentleyiniz.

hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df_h = df[["master_id","order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]
final_df_h["segment"] = segments
final_df_h.head()

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                   "order_num_total_ever_offline":["mean","min","max"],
                                   "customer_value_total_ever_offline":["mean","min","max"],
                                   "customer_value_total_ever_online":["mean","min","max"],
                                   "recency":["mean","min","max"],
                                   "tenure":["mean","min","max","count"]})