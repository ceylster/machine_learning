#### MEASUREMENT PROBLEMS(Olcum Problemleri) ####




### Rating Products(Urun Puanlama)
# Olasi faktorleri goz onunde bulundurarak agirlikli urun puanlama.

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating

#Uygulama:Kullanici ve Zaman Agirlikli Kurs Puani Hesaplama
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# (50+ Saat) Python A-Z:Veri Bilimi ve Machine Learning
# Puan: 4.8(4.764925)
# Toplam Puan: 4611
# Puan Yuzdeleri: 75, 20, 4, 1, <1
# Yaklasik Sayisal Karsiliklari: 3458, 922, 184, 46, 6

df = pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/course_reviews.csv")

df.head()

df.shape

# rating dagilimi
df["Rating"].value_counts()

df["Questions Asked"].value_counts()

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

df.head()

# Average

df["Rating"].mean()

# Memnuniyet trendi onemlidir. Takip etmekte fayda vardir.

# Time-Based Weigted Average(Puan zamanlarina gore agirlikli ortalama)

df.info()

# Timestamp sutununun tipini zaman degiskenine cevirmeliyiz
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Timestamp"].max()

current_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (current_date - df["Timestamp"]).dt.days

df[df["days"] <= 30]

df.loc[df["days"] <= 30, "Rating"].mean()

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

# Buradan kursun memnuniyetinde son zamanlarda bir artis oldugunu goruruz.

df.loc[df["days"] > 180, "Rating"].mean()

# agirlikli ortalama

df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[df["days"] > 180, "Rating"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[dataframe["days"] > 180, "Rating"].mean() * w4 / 100


# User-Based Weighted Average

df.head()

df.groupby("Progress").agg({"Rating": "mean"})

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

user_based_weighted_average(df, 20, 24, 26, 30)

# Weighted Rating

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return (time_based_weighted_average(dataframe) * time_w/100 +
            user_based_weighted_average(dataframe) * user_w/100)

course_weighted_rating(df)

### Sorting Products(Urun Siralama)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/product_sorting.csv")
print(df.shape)

df.sort_values("rating", ascending=False)

# Sorting by Comment Count or Purchase Count

df.sort_values("purchase_count", ascending=False)

df.sort_values("commment_count", ascending=False)

# Sorting by Rating, Comment and Purchase

df.describe().T

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])

(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return(dataframe["comment_count_scaled"] * w1 / 100 +
           dataframe["purchase_count_scaled"] * w2 / 100 +
           dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score",ascending=False)

# Bayesian Average Rating Score

# Puan dagilimlarinin uzerinden agirlikli bir sekilde olasiliksal ortalama hesabi yapar.

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5-Star Rating

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 -(1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1) * (n[k] + 1) / (N + K)
        second_part += (k+1) * (k+1) *  (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                               "2_point",
                                                               "3_point",
                                                               "4_point",
                                                               "5_point"]]), axis=1)

df.sort_values("bar_score",ascending=False)

df[df["course_name"].index.isin([5, 1])]

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score(Sorting Products with 5 Star Ranked)
# - Hybrid Sorting: BAR Score + Diger Faktorler


# Hybrid Sorting: Bar Score + Diger Faktorler

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)

    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False)

# IMDB Movie Scoring & Sorting

df = pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/movies_metadata.csv", low_memory=False)

df = df[["title", "vote_average", "vote_count"]]
df.head()

# Vote Average'a Gore Siralama
df.sort_values("vote_average", ascending=False).head(20)

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)

df["vote_count_score"] = (MinMaxScaler(feature_range=(1, 10)).
                          fit(df[["vote_count"]]).
                          transform(df[["vote_count"]]))

# vote_average * vote_count

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)

# IMDB Weighted Rating
# v = ilgili filmin oy sayisi, m = gereken oy sayisi, r = ilgili filmin puani, c = genel butun kitlenin ortalamasi

#weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500)) * 8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500)) * 8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 6.85 + 1 = 7.85

M = 2500
C = df["vote_average"].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(20)

weighted_rating(7.400, 11444.000, M, C)
weighted_rating(8.100, 14075.000, M, C)
weighted_rating(8.500, 8358.000, M, C)
df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(20)

# Bayesian Average Rating Score
# Potansiyelli urunleri ortaya cikarir.

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1) * (n[k] + 1) / (N + K)
        second_part += (k+1) * (k+1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df = pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/imdb_ratings.csv")

df = df.iloc[0:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four",
                                                               "five", "six", "seven", "eight",
                                                               "nine", "ten"]]), axis=1)

# Up-Down Diff Score = (up ratings) - (down ratings)

#Review 1: 600 up 400 down total 1000
#Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600,400)

# Review 1 Score:
score_up_down_diff(5500,4500)

# Average Rating = (up ratings) / (all ratings)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600,400)
score_average_rating(5500,4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)

# Wilson Lower Bound Score

# Bize ikili interactionlar barindiran item product review'i scorelama imkani saglar.(like, unlike gibi)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)
wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)

# Case Study

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})

# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],x["down"]),axis=1)

# score_acerage_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

comments.sort_values("wilson_lower_bound", ascending=False)

# AB Testing
# Sampling(Orneklem): bir ana kitle icerisinden bu ana kitlenin ozelliklerini iyi tasidigi varsayilan bir alt kumedir.




















