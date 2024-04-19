# Content Based Filtering (Icerik Temelli Filtreleme)

# Urun iceriklerinin benzerlikleri uzerinden tavsiyeler gelistirilir.
# 1. Metinlerin Matematiksel Olarak Temsil Et (Metinleri Vektorlestir)
#    - Count Vector (Word Count)
#    - TF-IDF
# 2. Benzerlikleri Hesapla
# Count Vectorizer
# Adim 1: Essiz tum terimleri sutunlara, butun tum dokumanlari satirlara yerlestir.
# Adim 2: Terimlerin dokumanlarda gecme frekanslarini hucrelere yerlestir.

# TF-IDF
# Kelimelerin hem kendi metinlerinde hem de butun odaklandigimiz verideki gecme frekanslari uzerinden bir
# normalizasyon islemi yapar. Count Vektorde cikacak bazi yanliliklari giderir.
# Adim 1: Count Vectorizer'i hesapla.
# Adim 2: Term Frequency'i hesapla. (Terim frekanslarini hesapla)
# (t teriminin ilgili dokumandaki frekansi / dokumandaki toplam terim sayisi)
# Adim 3: IDF - Inverse Document Frequency'i Hesapla.
# (1 + ln((toplam dokuman sayisi+1)/(icinde t terimi olan dokuman sayisi+1))
# Adim 4: TF*IDF hesapla
# Adim 5: L2 Normalizasyonu yap.
# (Satirlarin kareleri toplaminin karekokunu bul, ilgili satirdaki tum hucreceri buldugun deger bol.)

# Project: Content Based Recımmender System (Icerik Temelli Tavsiye Sistemi)

# Film Overview'larina Gore Tavsiye Gelistirme

# 1. TF-IDF Matrisinin Olusturulmasi
# 2. Cosine Similalarity Matrisinin Olusturulmasi
# 3. Benzerliklere Gore Onerilerin Yapilmasi
# 4. Calisma Scriptinin Hazirlanmasi

# 1. TF-IDF Matrisinin Olustrulmasi

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("Miull/Projects_doc/the_movies_dataset/movies_metadata.csv", low_memory=False)
df.head()

df["overview"].head()

tfidf = TfidfVectorizer(stop_words='english')
# df[df["overview"].isnull()]
df["overview"] = df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(df["overview"])
# fit: uygular, transform donusturur.

tfidf.vocabulary_
tfidf_matrix.toarray()

# 2. Cosine Sim Calculator (Cosine Similarity Matrisinin Olusturulmasi)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape
cosine_sim[1]

# Benzerliklerine Gore Onerilerin Yapilmasi

indices = pd.Series(df.index, index=df["title"])

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep="last")]  # en sonda cekilen filmi tutmak istiyoruz. Coklama mi sorusu.

# indices["Cinderella"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

df["title"].iloc[movie_indices]


# Preparation of Working Script (Calisma Scriptinin Hazirlanmasi)

def content_based_recommender(title, cosine_sim, dataframe):
    # indexleri olusturma
    indices = pd.Series(df.index, index=df["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    #titlein indexini yakalama
    movie_index = indices[title]
    #title'a gore benzerlik skorlarini hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    #kendisi haric ilk 10 filmi gosterme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe["title"].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Holiday", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe["overview"] = dataframe["overview"].fillna("")
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim





