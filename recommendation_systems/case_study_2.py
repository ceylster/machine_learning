# Hybrid Recommender System

# Is problemi
# ID'si verilen kullanici icin item-based ve user-based recommender yontemlerini kullanarak 10 film onerisi yapiniz.
# Veriseti, bir film tavsiye hizmeti olan Movie Lens tarafından saglanmistir. Icerisinde filmler ile birlikte bu
# filmlere yapilan derecelendirme puanlarini barindirmaktadir. 27278 filmde 20000263 derecelendirme icermektedir.
# Bu veri seti ise 17 Ekim 2016 tarihinde olusturulmustur. 138493 kullanici ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri
# arasinda verileri icermektedir. Kullanicilar rastgele secilmistir. Secilen tum kullanicilarin en az 20 filme oy
# verdigi bilgisi mevcuttur.

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 500)


# Gorev 1: Veri Hazirlama
# Adim 1: movie, rating veri setlerini okutunuz.

movie = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/movie.csv")
rating = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/rating.csv")

# Adim 2: rating veri setine Id’lere ait film isimlerini ve turunu movie veri setinden ekleyiniz.

df = movie.merge(rating, how="left", on="movieId")

# Adim 3: Toplam oy kullanilma sayisi 1000'in altinda olan filmlerin isimlerini listede tutunuz ve veri setinden
# cikartiniz.

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["count"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

# pd.reset_option("^display")

# Adim 4: index'te userID'lerin sutunlarda film isimlerinin ve deger olarak ratinglerin bulundugu dataframe icin
# pivot table olusturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"],values="rating")

# Adim5: Yapilan tum islemleri fonksiyonlastiriniz.

def data_preparation(df):
    movie = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/movie.csv")
    rating = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/rating.csv")
    df = rating.merge(movie, on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

# Gorev 2: Oneri Yapilacak Kullanicinin Izledigi Filmlerin Belirlenmesi

# Adim 1: Rastgele bir kullanici id’si seciniz.

random_user = int(pd.Series(user_movie_df.index).sample(1).values)
# random_user = 28941
# 111424
# 28941
# Adim 2: Secilen kullaniciya ait gozlem birimlerinden olusan random_user_df adinda yeni bir dataframe olusturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adim 3: Secilen kullanicilarin oy kullandigi filmleri movies_watched adında bir listeye atayiniz.

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
# len(movies_watched) = 122

# Gorev 3: Ayni Filmleri Izleyen Diger Kullanicilarin Verisine ve Id'lerine Erisilmesi

# Adim 1: Secilen kullanicinin izledigi fimlere ait sutunlari user_movie_df'ten seciniz ve movies_watched_df
# adinda yeni bir dataframe olusturunuz.

movies_watched_df = user_movie_df[movies_watched]

# Adim 2: Her bir kullancinin secili user'in izledigi filmlerin kacini izledigi bilgisini tasiyan user_movie_count
# adinda yeni bir dataframe olusturunuz.

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Adim 3: Secilen kullanicinin oy verdigi filmlerin yuzde 60 ve ustunu izleyenlerin kullanıcı id’lerinden
# users_same_movies adinda bir liste olusturunuz.

user_same_movies = user_movie_count[user_movie_count["movie_count"] >= (len(movies_watched) * 60 / 100)]["userId"].tolist()

len(user_same_movies)

# Gorev 4: Oneri Yapilacak Kullanici ile En Benzer Kullanicilarin Belirlenmesi
# Adım 1: user_same_movies listesi icerisindeki secili user ile benzerlik gosteren kullanicilarin id’lerinin
# bulunacagi sekilde movies_watched_df dataframe’ini filtreleyiniz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(user_same_movies)],random_user_df[movies_watched]])

# Adim 2: Kullanicilarin birbirleri ile olan korelasyonlaririn bulunacagi yeni bir corr_df dataframe’i olusturunuz.

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
#corr_df.dtypes

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

# Adim 3: Secili kullanici ile yuksek korelasyona sahip (0.65’in uzerinde olan) kullanicilari filtreleyerek top_users
# adında yeni bir dataframe olusturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

# Adim 4: top_users dataframe’ine rating veri seti ile merge ediniz.

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/rating.csv")

top_users_ratings = top_users.merge(rating, how="inner")

top_users_ratings.drop("timestamp", axis=1, inplace=True)

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
# Gorev 5: Weighted Average Recommendation Score'un Hesaplanmasi ve Ilk 5 Filmin Tutulmasi

# Adim 1: Her bir kullanicinin corr ve rating degerlerinin carpimindan olusan weighted_rating adinda yeni bir
# degisken olusturunuz.

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

# Adim 2: Film id’si ve her bir filme ait tum kullanicilarin weighted rating’lerinin ortalama degerini iceren
# recommendation_df adinda yeni bir dataframe olusturunuz.

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
# Adim 3: recommendation_df icerisinde weighted rating'i 2.2'den buyuk olan filmleri seciniz ve weighted rating'e
# gore siralayiniz.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values(by="weighted_rating", ascending=False)

# Adim 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seciniz

movie = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/movie.csv")

movies_to_be_recommend.merge(movie[["movieId", "title"]])




# Gorev 1: Kullanicinin izledigi en son ve en yuksek puan verdigi filme gore item-based oneri yapiniz.

# Adim 1: movie, rating verisetlerini okutunuz.

rating = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/rating.csv")
movie = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/movie.csv")

# Adim 2: Secili kullanicinin 5 puanverdigi filmlerden puani en guncel olan filmin id'sinin aliniz.

most_current_movies = rating[(rating["userId"] == random_user) & (rating["rating"] == 5)]. \
                          sort_values(by="timestamp", ascending=False)["movieId"].values[0]

# Adim 3: User based recommendation bolumunnde olusturulan user_movie_df dataframe’ini secilen film id’sine
# gore filtreleyiniz.

movie_df = movie[movie["movieId"] == most_current_movies]["title"].values[0]
movie_df = user_movie_df[movie_df]

# Adim 4: Filtrelenen dataframe’i kullanarak seçili filmle diger filmlerin korelasyonunu bulunuz ve siralayiniz.

movie_to_recommend = user_movie_df.corrwith(movie_df).sort_values(ascending=False)[1:6]