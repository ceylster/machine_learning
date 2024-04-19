import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option('display.width', 1000)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Sampling(Ornekleme)

populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()

np.random.seed(115)

orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()

# Orneklem sayisi arttikca ortalamalari populasyonun ortalamasina yakinsar. Orneklemden bir genelleme yapma cabasi
# vardir.

# Descriptive Statistics (Betimsel İstatistikler)

df = sns.load_dataset("tips")
df.describe().T

# Confidence Intervals(Guven Araligi)

# Anakütle parametresinin tahmini degerini(istatistik) kapsayabilecek iki sayidan olusan bir aralik bulmasidir.
# (n=100 ise olasi alabilecegin 100 orneklemden 95nin ortalamasi bu aralikta olacaktir..(ornek icin gecerli))

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# Restorantin musterilerinin odedegi hesap ortalamalari istatistiki olarak yuzde 95 guven ile
# (18.663331704358473, 20.908553541543167) degerleri arasindadir. Yuzde 5 de hata payimiz vardir.

sms.DescrStatsW(df["tip"]).tconfint_mean()
# Restorantin musterilerinin verecegi bahsis ortalamalari istatistiki olarak yuzde 95 guven ile
# (2.8237993062818205, 3.172758070767359) degerleri arasindadir. Yuzde 5 de hata payimiz vardir.

# Correlation (Korelasyon)

# Degiskenler arasindaki iliski, bu iliskinin yonu ve siddeti ile ilgili bilgiler saglayan istatiksel bir yontemdir.
# -1 ile 1 arasinda deger alir. 1 Mukemmel Pozitif Korelasyon -1 Mukemmel Negatif Korelasyon 0 ise Korelasyon yok
# demektir.

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show()

df["tip"].corr(df["total_bill"])
# Toplam hesap ile odenen bahsis arasinda pozitif yonlu orta siddetli bir iliski vardir. Odenilen hesap miktari
# arttikca bahsisin de artacagini ifade edebiliriz.

# Hypothesis Testing(Hipotez Testleri)
# Bir inanisi, bir savi test etmek icin kullanilan istatistiksel yontemlerdir. Grup karsilastirmalarinda temel amac
# olasi farklarinin sans eseri ortaya cikip cikmadigini gostermeye calismaktir.

# AB Testing (Bagimsiz Iki Orneklem T Testi)

# Iki grup arasinda karsilastirma yapilmak istenildiginde kullanilir. A kontrol grubu B deney grubudur. Iki grup da
# ayri ayri NORMAL dagilmasi gerekmektedir. Iki grubun varyanslarinin homojenligi varsayimidir.
# Varyanslarin homojenligi demek iki grubun dagilimlarinin birbirlerine benzer olup olmamasi demektir.
# 1-) Hipotezleri kur.
# 2-) Varsayimlari incele.
# 3-) P-valueya bakarak yorum yap.
# thesap > ttablo H0 red. P-value < 0.05 H0 red.

# 1-) Hipotezleri kur.
# 2-) Varsayimlari Kontrolu
#    - 1. Normallik Varsayimi
#    - 2. Varyans Homojenligi
# 3-) Hipotezin Uygulamasi
#    - 1. Varsayimlar saglaniyorsa bagimsiz iki orneklem t testi (parametrik test)
#    - 2. Varsayimlar saglanmiyorsa mannwhitneyu testi (non-parametrik test)
# 4-) p-value degerlerine gore sonuclari yorumla
# Not:
# - Normallik saglanmiyorsa direkt 2 numara. Varyans homojenligi saglanmiyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi oncesi aykiri deger incelemesi ve duzeltmesi yapmak faydali olabilir.

# AB testi problemimiz iki grup ortalamasina yonelikse bu adimlari kullanarak islemlerimizi gerceklestirebiliriz.

# Uygulama 1: Sigara icenler ile icmeyenlerin hesap ortalamalari aarasinda istatistiki olarak anlamli bir
# farklilik var midir?

df = sns.load_dataset("tips")

df.groupby("smoker").agg({"total_bill": "mean"})

# 1. Hipotezi kur.
# H0: M1 = M2
# H1: M1 != M2

# 2. Varsayim Kontrolu
# Normallik Varsayimi
# Varyans Homojenligi

# H0: Normal dagilim varsayimi saglanmaktadir.
# H1:...saglanmamaktadir.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 ise H0 Red. (Normal dagilim varsayimi saglanmamaktadir.(1. grup yani sigara icenler icin)

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugundan H0 reddedilir. (Normal dagilim varsayimi saglanmamaktadir.)

# O halde nunparametrik bir test kullanmamiz gerekir.

# Varyans Homojenligi Varsayimi

# H0: Varyanslar homojendir.
# H1: Varyanslar homojen degildir.

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugundan H0 reddedilir.(Varyanslar Homojen degildir.)

# 3. ve 4. Hipotezin Uygulanmasi

# 1. Varsayimlar saglaniyorsa bagimsiz iki orneklem t testi (parametrik test)
# 2. Varsayimlar saglanmiyorsa mannwhitneyu testi (non-parametrik test)

# -------------------------------EGER SAGLANSAYDİ------------------------
# Varsayimlar saglaniyorsa bagimsiz iki orneklem t testi (parametrik test)

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
# (Normallik varsayimi saglaniyor, varyans homojenligi varsayimi saglanmiyorsa da beni kullanabilirsin
# equal_var = False olmalidir. Arkada Welch testini yapar.

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value > 0.05 oldugundan H0 reddedilmez. Istatiksel olarak aralarinda anlamli bir fark yoktur.

# -----------------------------------------------------------------------

# 1.2 Varsayimlar saglanmiyorsa mannwhitneyu testi (non-parametrik test)
# mannwhitneyu testi non parametrik ortalama kiyaslama medyan kiyaslama testidir.

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value > 0.05 oldugundan H0 reddedilemez. Yani (M1 = M2) ortalamalari arasinda anlamli bir farklilik yoktur.
# H1 ile ilgili bir yorumda bulunamayiz. Yaptigimiz yorum H0 ile ilgilidir.


# Uygulama 2: Kadin ve Erkek yolcularin yas ortalamasi arasinda istatiksel olarak anlam farki midir?

df = sns.load_dataset("titanic")
df.groupby("sex").agg({"age": "mean"})

# 1. Hipotezleri kur:
# H0: M1 = M2 (Kadin ve Erkek yolcularin yas ortalamasi arasinda istatiksel olarak anlam fark yoktur.)
# H1: M1 != M2 (vardir)

# 2. Varsayimlari Incele

# Normallik varsayimi
# H0: Normal dagilim varsayimi saglanmaktadir.
# H1: ..saglanmamaktadir.

test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugundan H0 reddedilir.

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugundan H0 reddedilir.
# Iki grup icin de varsayim saglanmamaktadir.

# ----------------------------------------------------------------------------------------------------------------------
# OGRENMEK ICIN BAKIYORUZ. NONPARAMETRIK KISMA GECMEMIZ GEREKIRDI.
# Varyans Homojenligi
# H0: Varyanslar homojendir.
# H1: Varyanslar Homojen degildir.

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                                    df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value degeri > 0.05 oldugundan H0 reddedilemez. Varyanslar homojendir.
# ----------------------------------------------------------------------------------------------------------------------

# Varsayimlar saglanmadigi icin nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value degeri < 0.05 oldugundan dolayi H0 reddedilir. Kadin ve erkek yolcularinin arasinda gozlemledigimiz fark
# istatiksel olarak da vardir.

# Uygulama 3: Diyabet Hastasi olan ve olmayanlarin yaslari ortalamasi arasinda istatistiki olarak anlamli bir
# farklilik var midir?

df = pd.read_csv("Miull/Projects_doc/diabetes.csv")

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2 (Diyabet Hastasi olan ve olmayanlarin yaslari ortalamasi arasinda istatistiki olarak
#              anlamli bir fark yoktur)
# H1: M1 != M2 (... vardir)

# 2. Varsayimlari Incele

# Normallik Varsayimi (H0: Normal dagilim varsayimi saglanmaktadir.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test stat = %.4f, p-value = %.4f " % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test stat = %.4f, p-value = %.4f " % (test_stat, pvalue))

# p < 0.05 oldugundan H0 red normallik varsayimi saglanmamaktadir.

# Normallik varsayimi saglanmadigi icin nonparemetriktir.(nonparametrik medyan kiyasi da olabilir.)

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test stat = %.4f, p-value = %.4f " % (test_stat, pvalue))

# p-value < 0.05 oldugu icin H0: M1 = M2 (Diyabet Hastasi olan ve olmayanlarin yaslari ortalamasi arasinda
# istatistiki olarak anlamli bir fark yoktur) reddedilir. İstatsikti olarak anlamli bir farklilik vardir.

# Is Problemi: Kursun buyuk cogunlugunu izleyenler ile izlemeyenlerin puanlari birbirinden farkli mi?

# H0: M1 = M2 (Iki grup ortalamalari arasinda istatistiki olarak anlamli fark yoktur.)
# H1: M1 != M2 (vardir.)

df = pd.read_csv("Miull/Projects_doc/course_reviews.csv")

df[(df["Progress"] < 25)]["Rating"].mean()


df[(df["Progress"] < 25)]["Rating"].mean()

df[(df["Progress"] < 10)]["Rating"].mean()

test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p < 0.05 oldugundan H0 red normallik varsayimi saglanmamaktadir.

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])
print("Test stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugu icin reddedilir. İstatsikti olarak anlamli bir farklilik vardir.

# AB Testing (Iki Orneklem Oran Testi)

# Iki Grup Oran Karsilastirmasi (Iki Orneklem Oran Testi)
# Iki oran arasinda karsilastirma yapmak icin kullanilir.

# H0: p1 = p2 (Yeni tasarimin donusum orani ile eski tasarimin donusum orani arasinda istatiksel
# olan anlamli farklilik yoktur.)
# H1: p1 != p2 (vardir)

basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)

# p-value(0.0001532232957772221) < 0.05 oldugu icin reddedilir. İstatsikti olarak anlamli bir farklilik vardir.


# Uygulama: Kadin ve Erkeklerin hayatta kalma oranlari arasinda istatistiki olarak anlamli bir fark var midir?

# H0: p1 = p2 (Kadin ve Erkeklerin hayatta kalma oranlari arasinda istatistiki olarak anlamli bir fark yoktur.)

# H1: p1 != p2 (vardir)

df = sns.load_dataset("titanic")

df.groupby("sex").agg({"survived": "mean"})

df.loc[df["sex"] == "female", "survived"].mean()
df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
female_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, female_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                           df.loc[df["sex"] == "male", "survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 oldugundan dolayi H0 reddedilir. Yani Kadin ve Erkek hayatta kalma oranlari arasinda istatistiki
# olara bir fark vardir.

# ANOVA - Analysis of Variance

# Ikiden fazla grup ortalamasini karsilastirmak icin kullanilir.
# H0: M1 = M2 = M3
# H1: Eşit degillerdir.

# significance = p-valeu

df = sns.load_dataset("tips")
df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur.

# H0: m1 = m2 = m3 = m4 (Grup ortalamalari arasinda fark yoktur.)
# H1: En az biri farklidir.

# 2. Varsayim kontrolu

# Normallik varsayimi
# Varyans homojenligi varsayimi

# Varsayim saglaniyorsa one way anova
# Varsayim saglanmiyorsa kruskal

# H0: Normal dagilim varsayimi saglanmaktadir.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "p-value: %.4f" % pvalue)

# Dort deger icin de p-value < 0.05 oldugundan dolayi H0 reddedilir. Dolayisiyla hicbiri icin normallik varsayimi
# saglanmamaktadir.

# H0: Varyans homojeenligi varsayimi saglanmaktadir.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"],)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# h0 reddedilemez ancak normallik varsayimindan nanparametrik olmasina karar verdik.

# 3. Hipotez testi ve p-value yorumu

df.groupby("day").agg({"total_bill": ["mean", "median"]})

# varsayim saglansaydi paramaetrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# p-value < 0.05 oldugundan dolayi H0 reddedilir.

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# p-value < 0.05 oldugundan dolayi H0 reddedilir.

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

# Aykiri degerler oldugundan normallik varsayimi saglanmayabilir.
# Kombinasyonlara soktugumuzda hgata miktari takip edemeyecegimiz kadar artar. Bunlari ikili teste soktugumuzu zaman
# ucunu ayni anda kiyaslamiyoruz. Bunlardan dolayi ikili ikili kiayslamak teorik olarak mantikli degil. Hem grublar
# arasi hem de kendi aralarinda kiyaslamalari karsilastiriyor ANOVA.Her testin kendine özgü Tür 1 (yani doğru olduğu
# halde yanlışlıkla boş hipotezi reddetme olasılığı) hata olasılığı var (0,05) Yani her bir t testinde Tür 1 hatası
# yapmama olasılığı(0,95). Üç t testi olduğuna göre hata yapmama olasılığını üç kez kendisiyle
# çarpalım: 0,95 * 0,95 * 0,95 = 0,857 1 - 0,857 = 0,143. Yani Tür 1 hatası yapma olasılığı 0,05’ten 0,143’e yükseldi.
# Bu, kabul edilemez. Ikili kombinasyonlar yapamaiyoruz. Es zamanli degerlendiriyoruz.






















