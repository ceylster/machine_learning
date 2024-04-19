# Hierarchical Cluster Analysis (Hiyerarsik Kumeleme Analizi)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

df = pd.read_csv('Miull/Projects_doc/USArrests.csv', index_col=0)

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

# Uzaklik temelli yontemler kullandigimiz icin standartlastirmamiz gerekmektedir.

hc_average = linkage(df, "average")

plt.figure(figsize=(10, 5))
plt.title("Hiyerarsik Kumeleme Dendogrami")
plt.xlabel("Gozlem Birimleri")
plt.ylabel("Uzakliklar")
dendrogram(hc_average, leaf_font_size=10)

dendrogram(hc_average,
           leaf_font_size=10,
           truncate_mode="lastp",
           p=10,
           show_contracted=True)
plt.show()

# Hiyerarsik kumeleme yonteminin avantaji, gozlem birimlerine genelden bakma sansi tanir ve buna gore cesitli karar
# noktalarina dokunabilme sansi tanir.

# Kume Sayisini Belirleme


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=18)
plt.axhline(y=0.6, color='r', linestyle='--')
plt.show()

# Final Modelini Olusturmak

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv('Miull/Projects_doc/USArrests.csv', index_col=0)

df["hi_cluster_no"] = clusters

