# Unsupervised Learning (Gozetimsiz Ogrenme)

# Ilgilendigimiz problemde bir bagimli degisken bir hedef degisken yoksa yani ilgili gozlem birimlerinde meydana gelen
# gozlemlerde ortaya ne ciktigi bilgisi yoksa diger ifadeyle label, etiket yoksa bunlar gozetimsiz ogrenme
# problemleridir.

# K-Means (K-Ortalamalar)

# Amac gozlemleri birbirlerine olan benzerliklere gore kumelere ayirmaktir.
# Adim 1: Kume sayisi belirlenir.
# Adim 2: Rastgele k merkez secilir.
# Adim 3: Her gozlem icin k merkezlere uzakliklar hesaplanir.
# Adim 4: Her gozlem en yakin oldugu merkeze yani kumeye atanir.
# Adim 5: Atama islemlerinden sonra olusan kumeler icin tekrar merkez hesaplamasi yapilir.
# Adim 6: Bu islem belirlenen bir iterasyon adedince tekrar edilir ve kume ici hata kareler toplamlarinin toplaminin
# (total within-cluster variation) minimum oldugu durumdaki gozlemlerin kumelenme yapisi nihahi kume olarak secilir.

# pip install yellowbrick
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

df = pd.read_csv("Miull/Projects_doc/USArrests.csv", index_col=0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4).fit(df)
kmeans.get_params()
# n_clusters: kume sayisi

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_    # (SSD sum of squared distance)

# Optimum Kume Sayisini Belirleme

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farkli K Degerlerine Karsilik Uzaklik Artik Toplamlari")
plt.title("Optimum Kume Sayisi icin Elbow Yontemi")
plt.show()

# K means yontemi, hiyerarsik kumeleme yontemi gibi kumeleme yontemleri kullanilirken algoritmanin bize verdigi
# matematiksel referanslara gore, SSE'ye gore olan kume sayilarina direkt bakilarak is yapilmaz. Is bilgisiyle yapariz.
# Bu grafige gore bir belirleme yapacak olsaydik 5 sayisi dirseklemenin en fazla oldugu nokta olacakti muhtemelen. 5
# icin optimum nokta diyebilirdik. Ancak;

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

# Final Cluster'larin Olusturulmasi

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters = kmeans.labels_

df = pd.read_csv("Miull/Projects_doc/USArrests.csv", index_col=0)

df["clusters"] = clusters

df.head()

df[df["clusters"] == 1]

df[df["clusters"] == 5]

df.groupby("clusters").agg(["count", "mean", "median"])