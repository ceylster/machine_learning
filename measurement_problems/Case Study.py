# Rating Product & Sorting Reviews in Amazon

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# İs Problemi
# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasdr.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması
# ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların
# doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan
# etkileyeceğinden dolayı hem maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde
# e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak
# tamamlayacaktır.

df = pd.read_csv("Miull/Projects_doc/amazon_review.csv")

df.info()

# Görev1: Average Rating’i güncel yorumlara göre hesaplayınız ve var olan averagerating ile kıyaslayınız.
# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır. Bu görevde amacımız verilen
# puanları tarihe göre ağırlıklandırarak değerlendirmek. İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı
# puanın karşılaştırılması gerekmektedir.

# Adım1: Ürünün ortalama puanını hesaplayınız.

df["overall"].mean()

# Adım2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.


# df["day_diff"].describe([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]).T
# df["day_diff"].describe([0.25, 0.50, 0.75,]).T

df["dayss"] = pd.qcut(df["day_diff"], q=4, labels=["D", "C", "B", "A"])
df["dayss"].describe()

# 250, 450, 600

df.loc[df["day_diff"] <= 250, "overall"].mean()

df.loc[(df["day_diff"] > 250) & (df["day_diff"] <= 450), "overall"].mean()

df.loc[(df["day_diff"] > 450) & (df["day_diff"] <= 600), "overall"].mean()

df.loc[df["day_diff"] > 600, "overall"].mean()

# Adım3:Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
# bu sonuclara dayanarak urunun memnuniyet endeksinde olumlu bir artis gozlenmektedir.

df.loc[df["day_diff"] <= 250, "overall"].mean() * 28/100 + \
 df.loc[(df["day_diff"] > 250) & (df["day_diff"] <= 450), "overall"].mean() * 26/100 + \
 df.loc[(df["day_diff"] > 450) & (df["day_diff"] <= 600), "overall"].mean() * 24/100 + \
 df.loc[df["day_diff"] > 600, "overall"].mean() * 22/100

# Gorev2: Urun icin urun detay sayfasinda goruntulenecek 20 review’i belirleyiniz.
# Adım1: helpful_no degiskenini uretiniz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Adım2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.

def score_up_down_diff(up, down):
    return up - down
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)
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

# yeni bir kolon ekledigimiz icin axis 1
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

# Adım3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız.
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# Yuzde 95 guvenle yuzde 9775 helpdful yes alir.