# AB Testi ile Bidding Yontemlerinin Donusumunun Karsilastirilmasi

# İş Problemi


# Facebook kısa sure ocen mevcut "maximumbidding" adi verilen teklif verme turune alternatif
# olarak yeni bir teklif turu olan "average bidding"’i tanitti. Musterilerimizden biri olan bombabomba.com,
# bu yeni ozelligi test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla donusum
# getirip getirmedigini anlamak icin bir A/B testi yapmak istiyor.A/B testi bir aydir devam ediyor ve
# bombabomba.com simdi sizden bu A/B testinin sonuclarini analiz etmenizi bekliyor. Bombabomba.com icin
# nihai basari olcutu Purchase'dir. Bu nedenle, istatistiksel testler icin Purchasemetrigine odaklanilmalidir.

# Veri Seti Hikayesi

# Bir firmanın web site bilgilerini iceren bu veri setinde kullanicilari gordukeri ve tikladiklari
# reklam sayilar gibi bilgilerin yani sira buradan gelen kazanc bilgileri yer almaktadir.Kontrol ve Test
# grubu olmak uzere iki ayrı veri seti vardir. Bu veri setleriab_testing.xls excel’inin ayri sayfalarinda yer
# almaktadir. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# impression: Reklam goruntule sayisi
# Click: Goruntulenen reklama tiklama sayisi
# Purchase: Tiklanan reklamlar sonrasi satin alinan urun sayisi
# Earning: Satin alinan urunler sonrasi elde edilen kazanc

# Gorev1:  Veriyi Hazirlama ve Analiz Etme
# Adim1:  ab_testing_data.xlsx adli kontrol ve test grubu verilerinden olusan veri setini okutunuz. Kontrol ve test
# grubu verilerini ayri degiskenlere atayiniz.

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
pd.set_option("display.float_format", lambda x: "%.0f" % x)


# Adım2: Kontrol ve test grubu verilerini analiz ediniz.

df_control = pd.read_excel("Miull/Projects_doc/ab_testing.xlsx", "Control Group")
df_control.describe().T
df_control["Bidding"] = "Maximum_Bidding"

df_test = pd.read_excel("Miull/Projects_doc/ab_testing.xlsx", "Test Group")
df_test.describe().T
df_test["Bidding"] = "Average_Bidding"
# Adim3: Analiz isleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birlestiriniz.

df = pd.concat([df_control, df_test]).reset_index(drop = "index")

# Görev2: A/B Testinin Hipotezinin Tanimlanmasi
# Adım1: Hipotezi tanımlayınız.
# H0: M1 = M2 (arasinda istatistiki olarak anlamli bir fark yoktur.)
# H1: M1 != M2 (vardir.)

# Adım2: Kontrol ve test grubu icin purchase (kazanc) ortalamalarini analiz ediniz.
print(f'Kontrol grubu icin kazanc ortalamasi: {df.loc[(df["Bidding"] == "Maximum_Bidding")]["Purchase"].mean()}'
      f'\nTest grubu icin kazanc ortalamasi: {df.loc[(df["Bidding"] == "Average_Bidding")]["Purchase"].mean()}')

# Test ve kontrol grubu arasinda bir fark var ancak bunun sans eseri mi yoksa anlamli bir fark mi oldugunu inceleyelim.

# Görev3: Hipotez Testinin Gerceklestirilmesi
# Adım1: Hipotez testi yapilmadan once varsayim kontrollerini yapiniz. Bunlar Normallik Varsayimi ve Varyans
# Homojenligidir. Kontrol ve test grubunun normallik varsayimina uyup uymadigini Purchase degiskeni uzerinden
# ayri ayri test ediniz.

#  Varsayimlari Kontrolu
#   - 1. Normallik Varsayimi

test_stat, pvalue = shapiro(df.loc[df["Bidding"] == "Maximum_Bidding", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# pvalue degeri(0.5891) > 0.05 oldugundan H0 reddedilemez. Normal dagilim vardir.

test_stat, pvalue = shapiro(df.loc[df["Bidding"] == "Average_Bidding", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# pvalue degeri(0.1541) > 0.05 oldugundan H0 reddedilemez. Normal dagilim vardir.

#   - 2. Varyans Homojenligi
# H0: Varyanslar homojendir.
# H1: Varyanslar Homojen degildir.

test_stat, pvalue = levene(df.loc[df["Bidding"] == "Maximum_Bidding", "Purchase"],
                           df.loc[df["Bidding"] == "Average_Bidding", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# pvalue degeri(0.1083) > 0.05 oldugundan H0 reddedilemez. Yani varyanslar homojendir.

# Adım2: Normallik Varsayimi ve Varyans Homojenligi sonuclarina gore uygun testi seçiniz.
# Varsayimlar saglandigi icin bagimsiz iki orneklem t testi(parametrik testi) uygulayalim.

# Adım3: Test sonucunda elde edilen p_value degerini goz onunde bulundurarak kontrol ve test grubu satin alma
# ortalamalari arasinda istatistiki olarak anlamli bir fark olup olmadigini yorumlayiniz.

# Hem normallik hem de varyans homojenligi saglandigindan dolayi equal_var=True olacaktir.

test_stat, pvalue = ttest_ind(df.loc[df["Bidding"] == "Maximum_Bidding", "Purchase"],
                              df.loc[df["Bidding"] == "Average_Bidding", "Purchase"],
                              equal_var=True)

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value(0.3493) > 0.05 oldugundan H0 reddedilmez. Kontrol ve Test grubunun satin alma ortalamalari arasinda
# Istatistiki olarak anlamli bir fark yoktur.

# Gorev4: Sonuclarin Analizi

# Adım1: Hangi testi kullandiniz, sebeplerini belirtiniz.
# Hem normallik varsayimi hem de varyans homojenligi icin H0 reddedilemez oldugundan parametrik test olan bagimsiz
# iki orneklem t testini kullandik.

# Adım2: Elde ettiginiz test sonuclarina gore musteriye tavsiyede bulununuz.

# A/B testimiz sonucunda H0: M1 = M2 reddedilemeyeceginden dolayi yani Maximum_Bidding ve Minimum_Bidding degerlerimiz
# anlamli bir farkliliklari olmadiginda dolayi kullanilan yeni sistemin simdilik yetersiz oldugunu gorduk ve
# Average Bidding satin alinmamasi yonunde bir gorus bildirebiliriz. Ancak eldeki verilerin de az oldugu bunun icin
# daha fazla veriyle tekrar inceleyebilecegimizi soyleyebiliriz.

# Z ve T testi icin 30 kosul yeterdir.

