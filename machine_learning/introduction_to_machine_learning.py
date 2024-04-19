# Introduction to Machine Learning
# Bilgisayarlarin insanlara benzer sekilde ogrenmesini saglamak maksadiyla cesitli algoritma ve tekniklerin
# gelistirilmesi icin calisilan bilimsel calisma alanidir.


# Variable Types(Degisken Turleri)

# - Sayisal Degiskenler
# - Kategorik Degiskenler (Nominal(Siniflar arasinda fark yoktur.), Ordinal(Siniflar arasinda fark vardir.))
# - Bagimli Degisken(target, dependent, output, response)
# - Bagimsiz Degisken(feature, independent, input, column, predictor, explanatory)

# Learning Types(Ogrenme Turleri)

# - Supervised Learning (Denetimli Ogrenme) (Bagimli degisken yer aliyorsa denetimli ogrenme problemine sahiptir.)
# - Unsupervised Learning (Denetimsiz Ogrenme) (Labellar yoksa yani bagimli degisken yoksa denetimsiz ogrenmedir.)
# - Reinforcement Learning (Pekistirmeli Ogrenme)

# Problem Types (Problem Turleri)

# - Siniflandirma Problemleri: bagimli degisken kategorik
# - Regresyon Problemleri: bagimli degisken sayisal

# Model Basiri Degerlendirme Yontemi

# - MSE (Hata kareler ortalamasi)
# - RMSE (Hata kareler ortalamasi karekoku)
# - MAE (Mutlak ortalama hata)

# Siniflandirma Modellerinde Basari Degerlendirme

# Accuracy = Dogru Siniflandirma Sayisi / Toplam Siniflandirilan Gozlem Sayisi

# Model Validation (Model Dogrulama Yontemleri)

# - Holdout Yontemi (Sinama Test Yontemi)
# - K Fold Cross Validation (K-Katlli Capraz Dogrulama)
# Gozlem sayisi az oldugunda tum veri setini cross validation daha saglikli olur.

# Bias-Variance Tradeoff (Yanlilik - Varyans Degis Tokusu)

# Underfitting (Yuksek Yanlilik, dusuk varyans, modelin veriyi ogrenememsidir)
# Dogru Model (Dusuk yanlilik, dusuk varyans)
# Overfitting (Yuksek varyans, yani modelin veriyi ezberlemesidir. Biz verinin yapisini ogrenmesini istiyoruz.)

# Not: Asiri ogrenme egitim seti ve test seti hata degisimleri incelenir bu iki hatanin birbirinden ayrilmaya basladigi
# nokta itibariyle asiri ogrenme baslamistir.

# Model Karmasikligi: Modelin daha hassas tahmin yapabilmesi icin daha detaylandirilmasi islemidir.
