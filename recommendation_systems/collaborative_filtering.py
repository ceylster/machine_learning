# Collaborative Filterin (Is Birlikci Filtreleme)

# Item-Based Collaborative Filtering (Memory-Based)
# User-Based Collaborative Filtering (Memory-Based)
# Model-Based Collaborative Filtering

# Item-Based Collaborative Filtering
# Item benzerligi uzerinden oneriler yapilir.

# Project: Item-Based Recommendation System

# Adim 1: Veri Setinin Hazirlanmasi
# Adim 2: User Movie Df'inin Olusturulmasi
# Adim 3: Item-Based Film Onerilerinin Yapilmasi
# Adim 4: Caalisma Scriptinin Hazirlanmasi

# Adim 1: Veri Setinin Hazirlanmasi

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 500)
movie = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/rating.csv')

df = movie.merge(rating, on='movieId', how='left')

# Adim 2: User Movie Df'inin Olusturulmasi

df["title"].nunique()

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["count"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

common_movies["title"].nunique()
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values="rating")

# Adim 3: Item-Based Film Onerilerinin Yapilmasi

movie_name = "Matrix, The (1999)"

movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

def chech_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]


# Adim 4: Calisma Scriptinin Hazirlanmasi

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, on='movieId', how='left')
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values="rating")
    return user_movie_df

def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender('Terminator 2: Judgment Day (1991)', user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)


# User-Based Collaborative Filtering (Kullanici Tabanli Is Birlikci Filtreleme)

# Kullanici(User) benzerlikleri uzerinden oneriler yapilir.

# Project: User-Based Recommendation System

# Adim 1: Veri Setinin Hazirlanmasi
# Adim 2: Oneri Yapilacak Kullanicinin Izledigi Filmlerin Belirlenmesi
# Adim 3: Ayni Filmleri Izleyen Diger Kullanicilarin Verisine ve Id'lerine Erismek
# Adim 4: Oneri Yapilacak Kullanici ile En Benzer Davranisli Kullanicilarin Belirlenmesi
# Adim 5: Weighted Average Recommendation Score'un Hesaplanmasi
# Adim 6: Calismanin Fonksiyonlastirilmasi

# Adim 1: Veri Setinin Hazirlanmasi
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, on='movieId', how='left')
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Adim 2: Practical to Bring Watched Movies

random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]

len(movies_watched)

# Adim 3: Other Users Watching The Same Movies (Ayni Filmleri Izleyen Diger Kullanicilar)

movies_watched_df = user_movie_df[movies_watched]

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]


# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


# Adim 4: Determination of Similarity (Oneri Yapilacak Kullanici ile En Benzer Davranisli Kullanicilarin Belirlenmesi)

# Bunun icin 3 adim gerceklestirecegiz:
# 1. Sinan ve diger kullanicilarin verilerini bir araya getirecegiz.
# 2. Korelasyon df'ini olusturacagiz.
# 3. En benzer kullanicilari (Top Users) bulacagiz.

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by="corr", ascending=False)

top_users.rename(columns={"user_id_2": "user_id"}, inplace=True)

rating = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/rating.csv')

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# Score Calculation (Weighted Average Recommendation Score'un Hesaplanmasi)

top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movie = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/movie.csv')

movies_to_be_recommend.merge(movie[["movieId", "title"]])

# Adim 6: Functionalization (Calismanin Fonksiyonlastirilmasi)

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, on='movieId', how='left')
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=['userId'], columns=['title'], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('Miull/Projects_doc/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])


random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, score=4)

# Matrix Factorization (Matris Carpanlarina Ayirma)

# Bosluklari doldurmak icin userlar ve movieler icin car oldugu varsayilan latent featurelarin agirliklar var olan
# veri uzerinden bulunur ve bu agirliklar ile var olmayan gozlemler icin tahmin yapilir.

# - User-Item matrisini 2 tane daha az boyutlu matrise ayristirir.
# - 2 matristen User-Item matrisine gidisin latent factorler ile gerceklestigi varsayiminda bulunur.
# - Dolu olan gozlemler uzerinden latent factorlerin agirliklarini bulur.
# - Bulunan agirliklar ile bos olan gozlemler doldurulur.
# - Rating matrisinin iki factor matris carpimi (dot product) ile olustugu varsayilir.
# - Factor matrisler: user latent factors ve movie latent factors
# - Latent factors: Latent features, gizli faktorler ya da degiskenler
# - Kullanicilarin ve filmlerin latent featurekar icin skorar sahip oldugu dusunulur.
# - Bu agirliklar(skorlar) once var olan veri uzerinden bulunur ve sonra BLANK bolumler bu agirliklara gore doldururlur.
# - Var olan degerler uzerinde iteratif sekilde tum p ve q'lar bulunur ve sonra kullanilirlar.
# - Baslangicta rastgele p ve q degerleri ile rating matrisindeki degerler tahmin edilmeye calisilir.
# - Her iterasyonda hatali tahminler duzenlenerek rating matristeki degerlere yaklasilmaya calisilir.
# - Ornegin bir iterasyonda 5e 3 dendiyse sonrakinde 4 sonrakinde 5 denir.
# - Boylkece belirli bir iterasyon sonucunda p ve q matrisleri doldurulmus olur.
# - Var olan p ve qlara gore bos gozlemler icin tahmin yapilir.

# Gradient Descent (Gradyan Inis)
# Gradient Descent fonksiyon minimizasyonu icin kullanilan bir optimizasyon yontemidir.
# Gradyanin negatifi olarak tanimlanan en dik inis yonunde iteratif olarak parametre degerlerini guncelleyerek
# ilgili fonksiyonun minimım degerini verecek parametreleri bulur.

# Model-Based Collaborative Filtering: Matrix Factorization
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option("display.max_columns", None)

# Adim 1: Preparation of Data (Verinin Hazirlanmasi)
# Adim 2: Modelleme
# Adim 3: Model Tuning
# Adim 4: Final Model ve Tahmin

# Adim 1: Preparation of Data (Verinin Hazirlanmasi)

movie = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/movie.csv")
rating = pd.read_csv("Miull/Projects_doc/movie_lens_dataset/rating.csv")
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark  Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]

user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values="rating")

reader = Reader(rating_scale=(1, 5))     # Hesap yapabilmesi icin rate'lerin skalasini veriyoruz.

data = Dataset.load_from_df(sample_df[["userId",
                                       "movieId",
                                       "rating"]], reader)

# Adim 2: Modelling (Modelleme)

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)           # Hata kareler ortalamasinin karakoku

svd_model.predict(uid=1.0, iid=541, verbose=True)   # (user id:uid, item id: iid)
svd_model.predict(uid=1.0, iid=356, verbose=True)

sample_df[sample_df["userId"] == 1]

# Adim 3: Model Tuning

param_grid = {"n_epochs": [5, 10, 20],
              "lr_all": [0.002, 0.005, 0.007]}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1, joblib_verbose=True )
# (cv=capraz dogrulama. 3 katli, 2 parcasiyla model kur 1 parcasi ile test et 1. islem, diger 2 parcayla model kur
# diger kalan ile test et. en sonda 3unun ortalamasini al. mae: mutlak hata ortalamasi, n_jobs islemcileri full
# performansi ile kullan demek. joblib_verbose islemler gerceklesirken bana raporlama yap. )

gs.fit(data)

gs.best_score["rmse"]
gs.best_params["rmse"]

# Adim 4: Final ve Tahmin

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params["rmse"])   # ** arguman girmek icin kullaniriz.

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)
























