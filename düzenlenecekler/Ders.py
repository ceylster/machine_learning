#Aşağıdaki verilen maaslarin 3000 ve uzeri olanlarina yuzde 10 altına yuzde 23 olacak sekilde zam yapilacaktir. Uygun bir fonksiyon yaziniz.
import numpy
import pandas as pd

salaries = [1000, 1250, 2000, 3000, 4000, 4500, 5000]


def new_salary(salary, rate):
    return int(salary*rate/100 + salary)


for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary,10))
    else:
        print(new_salary(salary,20))


#Asagidaki sakilde string degistiren fonksiyon yazmak istiyoruz.

# Before: "hi my name is john and i am learning python"
# After: "Hi mY NaMe  iS JoHn aNd i aM LeArNiNg pYtHon"


def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("ali")

alternating("hi my name is john and i am learning python")


# Uygulama - Mülakat Sorusu

# divide_students fonksiyonunu yazınız.
# Cift indexte yer alan ogrencileri bir listeye aliniz.
# Tek indexte yer alan ogrencileri baska bir listeye aliniz.
# Fakat bu iki liste tek bir liste olarak return olsun.


students = ["John", "Mark", "Venessa", "Mariam"]

def divede_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

divede_students(students)


#Asagidaki sakilde string degistiren fonksiyon yazmak istiyoruz.

# Before: "hi my name is john and i am learning python"
# After: "Hi mY NaMe  iS JoHn aNd i aM LeArNiNg pYtHon"

def alternating_with_enumaerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating_with_enumaerate("hi my name is john and i am learning python")

#Gorev 1: Verilen degerlerin yapilarini inceleyiniz.

x = 8
type(x)

y = 3.2
type(3.2)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)

l = [1, 2, 3, 4]
type(l)

d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
type(d)

t = ("Machine Learning", "Data Science")
type(t)

s = {"Python", "Machine Learning", "Data Science"}
type(s)

#Gorev 2: Verilen string ifadenin tum harflerini buyuk harfe çeviriniz. Vırgul ve nokta yerine space koyunuz. Kelime kelime ayiriniz.

text = "The goal is to turn data into information, and information into insight"

a = text.upper()

b = a.replace(",", "")

c = b.split()

c

#Gorev 3: Verilen listeyi asagidaki adimlari uygulayiniz.


lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

len(lst)

lst[0]
lst[10]

lst[0:4]

lst.pop(8)
lst

lst.append("ali")
lst

lst.insert(8, "N")
lst

lst.pop(11)
lst

#Gorev 4: Verilen sozluk yapisina asagidaki adimlari uygulayiniz.

dict = {'Christian':["America", 18],
        'Daisy':["England", 12],
        'Antonio':["Spain", 22],
        'Dante':["Italy",25]}

dict.keys()

dict.values()

dict['Daisy'] = ["England", 13]
dict
dict['Daisy'][1] = 14

dict.update({'Ahmet': ["Turkey", 24]})


dict.pop('Antonio')
dict

#Gorev 5: Arguman olarak bir liste olan, listenin icerisindeki tek ve cift sayilari ayri listelere atayan ve bu listeleri return eden fonksyon yaziniz.

l = [2, 13, 18, 93, 22]

def func(list):
    even_number = []
    odd_number = []
    for i in list:
        if i % 2 == 0:
            even_number.append(i)
        else:
            odd_number.append(i)
    return even_number, odd_number

func(l)

#Gorev 6: Asagida verilen listede muhendislik ve tip fakultelerinde dereceye giren ogrencilerin isimleri bulunmaktadir.
# Sirasiyla ilk uc ogrenci muhendislik fakultesinin basari sirasini temsil ederken son uc ogrenci de
# tip fakultesi ogrenci sirasina aittir. Enumare kullanarak ogrenci derecelerini fakulte ozelinde yazdiriniz.

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

muhendislik_sirasi = []
tip_sirasi = []

for index, student in enumerate(ogrenciler):
    if index < 3:
        index += 1
        print("Muhendislik Fakultesi",index, ". ogrenci: ", student)
    else:
        index -= 2
        print("Tip Fakultesi",index, ". ogrenci: ", student)


#Gorev 7: Asagidaki 3 adet liste verilmistir. Listelerde sirasi ile bir dersin kodu, kredisi ve kontenan bilgilieri yer
#almaktadir. Zip kullanilarak ders bilgilerini bastiriniz.

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

list(zip(ders_kodu,kredi,kontenjan))

for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjani {kontenjan} kisidir")
isim = "Ali"
yas = 25
print("Merhaba " + isim + " " + str(yas) + " yasindasin.")
print(f"Merhaba {isim} {str(yas)} yasindasin.")
#Gorev 8: Asagida 2 adet set verilmistir. Sizden istenilen eger 1. kume 2. kumeyi kapsiyor ise ortak elemanlarini eger
#kapsamiyor ise 2. kumenin 1. kumeden farkini yazdiracak fonksiyonu tanımlamanız beklenmektedir.


kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def kume(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

kume(kume1,kume2)
kume(kume2,kume1)

#Uygulama

#Amac key'i string, value'su asagidaki gibi bir liste olan sozluk olusturmak ve sadece sayisal degiskenler icin yapmak istiyoruz.
#Sadece sayisal degiskenler icin yapmak istiyoruz.

# {"total": ["mean", "min", "max", "var"],
#  "speeding" : ["mean", "min", "max", "var"],
#  "alcohol" : ["mean", "min", "max", "var",],
#  "not_distracted" : ["mean", "min", "max", "var",],
#  "no_previous" : ["mean", "min", "max", "var",],
#  "ins_premium" : ["mean", "min", "max", "var",],
#  "ins_losses" : ["mean", "min", "max", "var",]}

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
num_cols = [col for col in df.columns if df[col].dtype != "O"]
soz = {}
agg_list = ["mean", "min", "max", "var",]

for col in num_cols:
    soz [col] = agg_list

#kisa yol

new_dict = {col : agg_list for col in num_cols}

df[num_cols].agg(new_dict)

numbers = [12 ,75 ,150 ,180 ,145 ,525 , 50]


largest=numbers[0]
for large in numbers:
    if large > largest:
        largest=large
print(largest)

for i in range(1,13):
    if i % 2 != 0:
        continue
    print(i)


numbers = [12, 75, 150, 180, 145, 525, 50]

for i in numbers:
    if i > 500:
        break
    elif i > 150:
        continue
    elif i % 5 == 0:
        print(i)

list = [10, 20, 30, 40, 50,]

list[::-1]

a = [12, 35, 9, 56, 24]

def swaplist(new_list):
    size = len(new_list)
    temp = new_list[0]
    new_list[0] = new_list[size - 1]
    new_list[size - 1] = temp
    return new_list

swaplist(a)

new = []
string = "geeks quiz practice code"
s = string.split()
t = s[::-1]
for i in t:
    new.append(i)
print(" ".join(new))

def get_even_numbers(numbers):
    return [num for num in numbers if num % 2 == 0]
    print("Even numbers:", get_even_numbers(numbers))

a = [1,2,3,4,5,6,7,8,9,10]

get_even_numbers(a)

for i in range(1,100):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

#Description: Write a function that takes two strings as input and returns True if they are anagrams (contain the same
# characters with the same frequency), False otherwise. This exercise involves comparing the characters and their
# frequencies in two strings to determine if they are anagrams.
# You will practice string manipulation, sorting, and comparison.

def is_anagram(s1, s2):
    s1 = s1.replace(" ","").lower()
    s2 = s2.replace(" ","").lower()
    return sorted(s1) == sorted(s2)

string1 = "Listen"
string2 = "Silent"

if is_anagram(string1,string2):
    print("The strings are angrams")
else:
    print("The strings are not angrams")


#prime numbers:
def is_prime(n):
    if n <= 1:
       return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

#Write a Python program to print all even numbers from a given list of numbers in the same order and stop printing any
# after 237 in the sequence.

numbers = [386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345,
           399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217,
           815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717,
           958,743, 527]

for i in numbers:
    if i == 237:
        print(i)
        break
    elif i % 2 == 0:
        print(i)

#Write a Python program that calculates the area of a circle based on the radius entered by the user.
r = float()
from math import pi
def circlearea(r):
    area = pi * r ** 2
    print("this area is", area)

circlearea(2)

def type_func(x):
    if type(x) is list:
        print("x is a list")
    elif type(x) is tuple:
        print("x is a tuple")
    elif type(x) is set:
        print("x is a set")
    else:
        print("Neither a list nor a set nor a tuple")


a = (1, 3, 4 ,6)
type_func(a)

def divisable(x,y):
    x,y != 0
    if x % y == 0:
        print("x is divisable for y")
    elif y % x == 0:
        print("y is divisable for x")
    else:
        print("Neither x nor y is divisable for eachother.")

divisable(114, 0)

num_list = [45, 55, 60, 37, 100, 105, 220]

list(filter(lambda x: x % 15 == 0,num_list))

nums = [34, 1, 0, -23, 12, -88]

new_nums = list(filter(lambda x: x >= 0, nums))

new_nums.sort()

#List Comprehension

#Görev 1:  ListComprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin
# isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

# num_cols = [col for col in df.columns if df[col].dtype != "O" ]

num_cols = ["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

#Görev 2:  List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin
#isimlerinin sonuna "FLAG" yazınız.

flag_cols = [col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns]

#Görev 3:List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin
#isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

og_list = ["abbrev", "no_previous"]

out_ogg_cols = [col for col in df.columns if col not in og_list]
new_df = df[out_ogg_cols]
new_df_2 = df[[col for col in df.columns if col not in og_list]]

import matplotlib.pyplot as plt

notlar = [68, 74, 82, 90, 78, 85, 92, 88, 76, 61, 79, 73, 89, 81, 72, 95, 70, 83, 77, 75]

plt.hist(notlar, bins=10, edgecolor='r', alpha=0.7)
plt.xlabel('Notlar')
plt.ylabel('Frekans')
plt.title('Sınav Notları Dağılımı')
plt.show()

### Pandas Alistirmalari ###

#1 Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df.shape #891 gozlemimiz ve 15 tane de degiskenimiz vardir.

#Görev 2:Titanic verisetindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()

#Görev3:Her bir sutuna ait unique değerlerin sayısını bulunuz.
df.nunique()

#Görev4:pclass değişkeninin unique değerlerinin sayısını bulunuz.
df["pclass"].nunique()

#Görev5:pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
df[["pclass", "parch"]].nunique()
df.loc[:, ["pclass", "parch"]].nunique()


#Görev6:embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df["embarked"].info()
df["embarked"].dtypes
df["embarked"].astype("category")
df["embarked"] = df["embarked"].astype("category")

df["embarked"] = pd.Categorical(df["embarked"]) #farkli bir yolu...
df["embarked"].dtypes

#Görev7:embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == "C"].head()

#Görev8:embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != "S"].head()

#Görev9:Yaşı 30'dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df.loc[(df["age"] < 30) & (df["sex"] == "female")]
df[(df["age"] < 30) & (df["sex"] == "female")]


#Görev10:Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df.loc[(df["fare"] > 500) | (df["age"] > 70)]
df[(df["fare"] > 500) | (df["age"] > 70)].head()

#Görev11:Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()


#Görev 12:who değişkenini dataframe’den çıkarınız.

df.drop("who", axis=1)
#inplace=True kalicilik saglar

#Görev13:deck değiskenindeki boş değerleri deck değişkenin en çok tekrar eden değeri(mode) ile doldurunuz.
df["deck"].value_counts()
df["deck"].mode()
type(df["deck"].mode()) #tipine gore bir gidisat belirtecegiz. Seri oldugunu goruyoruz ciktida.
df_deck = df["deck"].fillna(df["deck"].mode()[0])
df_deck.isnull().sum()


m = df["deck"].mode()
for index in df["deck"][df["deck"].isnull()].index:
    df["deck"][index] = m


# [0] anlamı nedir? 1. sıradaki mod degeri icin [0] degerini yaziyoruz.

#Görev14:age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"].median()
df["age"].fillna(df["age"].median())

median = df["age"].median()
df.loc[df["age"].isnull(), "age"] = median
df["age"].value_counts()


#Görev15:survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({"survived" : ["sum", "count", "mean"]})

#Görev16:30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

#Görev17:Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df= sns.load_dataset("tips")
df.head()

#Görev18:Time değişkeninin kategorilerine (Dinner,Lunch) göre total_bill değerlerinin toplamını, min,
#max ve ortalamasını bulunuz.

df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

#Görev19:Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

df.pivot_table("total_bill", "day", "time", aggfunc=["min", "max", "mean", "sum"])


#Pivot table ile Groupbby'in kullanim farki nedir?

#Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e
#göre toplamını, min, max ve ortalamasını bulunuz.

df.loc[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby(["day"]).agg({"total_bill" :["sum", "min", "max", "mean"],
                                                                    "tip" :["sum","min", "max", "mean"]})

#Görev21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?(loc kullanınız)

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()

#Görev22:total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği total bill ve tip'in
#toplamını versin.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

#Görev23:total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir
#dataframe'e atayınız.

df["total_bill_tip_sum"].sort_values(ascending=False).head(30)


df = df.sort_values("total_bill_tip_sum", ascending=False).head(30)

df.sort_values(by="total_bill_tip_sum", ignore_index=True, ascending=False, inplace=True)

#Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df=pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/persona.csv")

#Soru1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

df=pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/persona.csv")
df.head()

#Soru 2:Kaç unique SOURCE vardır? Frekansları nedir?

df["SOURCE"].nunique()
df["SOURCE"].unique()
df["SOURCE"].value_counts()

#Soru3:Kaç unique PRICE vardır?
df["PRICE"].nunique()

#Soru 4:Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df["PRICE"].value_counts()

#Soru 5:Hangi ülkeden kaçar tane satış olmuş?

df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()

#Soru 6:Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("COUNTRY").agg({"PRICE": "sum"})
df.pivot_table("PRICE","COUNTRY",aggfunc="sum")

#Soru 7:SOURCE türlerine göre satış sayıları nedir?

df["SOURCE"].value_couts()
df.groupby("SOURCE").agg({"PRICE": ["sum", "count"]})

#Soru 8:Ülkelere göre PRICE ortalamaları nedir?

df.groupby("COUNTRY").agg({"PRICE": "mean"})

#Soru 9:SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby("SOURCE").agg({"PRICE": "mean"})

#Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

df.pivot_table("PRICE","COUNTRY", "SOURCE") (#on tanimli deger meandir.)

#Görev 2:  COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

#Görev 3:Çıktıyı PRICE’a göre sıralayınız.

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})

agg_df = agg_df["PRICE"].sort_values(ascending=False)

#Görev 4:Indekste yer alan isimleri değişken ismine çeviriniz

agg_df = agg_df.reset_index()

#Görev 5:Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, 70], labels=["0_18", "19_23", "24_30", "31_40", "41_70"])


#Görev6:Yeni seviye tabanlı müşterileri (persona) tanımlayınız
#- Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz
#- Yeni eklenecek değişkenin adı: customers_level_based
#- Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based
#değişkenini oluşturmanız gerekmektedir.
#for row in agg_df.values:
#    print(row)

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper()
                                   for row in agg_df.values]

#agg_df["customers_level_based"] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGECAT']].apply(lambda x: '_'.join
#                                                                                        ([str(y).upper() for y in x]), axis = 1)
#def find(row):
#     return '_'.join(str(row[col]).upper() for col in agg_df.columns if col not in ['PRICE','AGE'])
#agg_df['customers_level_based']=agg_df.apply(find,axis=1)



agg_df["customers_level_based"].value_counts()

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()

#Görev 7:Yeni müşterileri(personaları) segmentlere ayırınız.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "min", "max", "sum"]})

#Görev 8:Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.
#33 yaşında ANDROID kullanan bir Türk kadını hangi segmenteaittir ve ortalama ne kadar gelir kazandırması beklenir

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]["SEGMENT"].unique()
agg_df[agg_df["customers_level_based"] == new_user]["PRICE"].unique().mean()


#35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]["SEGMENT"].unique()
agg_df[agg_df["customers_level_based"] == new_user]["PRICE"].unique().mean()


### FLO RFM Analizi

#Adım 1:flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz.
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


df_ = pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/flo_data_20k.csv")

df = df_.copy()

# Adım2:Veri setinde
#       a. İlk 10 gözlem,
#       b. Değişken isimleri,
#       c. Betimsel istatistik,
#       d. Boş değer,
#       e. Değişken tipleri, incelemesi yapınız

df.head(10)
df.info()

df.columns

df.shape
df["master_id"].nunique()
df.describe().T
df["order_channel"].unique()
df["order_channel"].value_counts()
df["interested_in_categories_12"].unique()

df.isnull().sum()

df.info()

#Adım3:Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
#Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

#toplam alisveris miktari
df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

#toplam tutar miktari
df["custumer_value_total_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head(10)

#Adım4:Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

#date_columns = df.columns[df.columns.str.contains("date")]
#df[date_columns] = df[date_columns].apply(pd.to_datetime)


#date_columns = [col for col in df.columns if "date" in col]
#df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x))


#Adım5:Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.shape
df["master_id"].nunique()

df.groupby("order_channel").agg({"order_num_total_omnichannel" : ["sum", "mean", "count"],
                                 "custumer_value_total_omnichannel": ["sum", "mean", "count"]})

df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_omnichannel" : "sum",
                                 "custumer_value_total_omnichannel":"sum",})

#Adım6:En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df.sort_values("custumer_value_total_omnichannel", ascending=False).head(10)

#Adım7:En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df.sort_values("order_num_total_omnichannel", ascending=False).head(10)

#Adım8:Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_preparation(dataframe):
    dataframe["order_num_total_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["custumer_value_total_omnichannel"] = dataframe["customer_value_total_ever_offline"] + dataframe[ "customer_value_total_ever_online"]

    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])

    return dataframe

##Görev 2:RFM Metriklerinin Hesaplanması  ##

#Adım 1:Recency, Frequency ve Monetary tanımlarını yapınız.
#Recency:Son alisveris yaptigi gunden belirlenen tarihe kadar gecen sure
#Frequency:Siparis sikligi
#Moneytary:Toplam harcanan para

#Adım 2:Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız
#recency değerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz
df.sort_values("last_order_date").tail()
today_date = dt.datetime(2021, 6, 1)

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                   "order_num_total_omnichannel": "sum",
                                   "custumer_value_total_omnichannel": "sum"})


#OMER FARUK AYTUNC


# df["last_order_date"].max()
# analysis_date = dt.datetime(2021, 6, 2)
# df["recency"] = (analysis_date - pf["last_order_date"]).astype("timedelta64[D]")
# df["recency"] = (analysis_date - pd.to_datetime(df["last_order_date"])).dt.days
# rfm = df[["master_id", "recency", "total_order", "total_customer_value"]]
# rfm.columns = ["master_id", 'recency', 'frequency', 'monetary']

#SAID YILMAZ
# df["Recency"] = [(today_date - date).days for date in df["last_order_date"]]


rfm["order_num_total_omnichannel"] = rfm["order_num_total_omnichannel"].astype(int)

#Adım 4:Oluşturduğunuz metriklerin isimlerini  recency, frequency ve monetary olarak değiştiriniz.

rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()

#Görev3: RF Skorunun Hesaplanması
#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Adım 2: Bu skorları recency_score, frequency_scoreve monetary_scoreolarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), q=5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])

#Adım 3:recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm.dtypes

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

#Görev 4:RF Skorunun Segment Olarak Tanımlanması
#Adım 1:Oluşturulan RF skorları için segment tanımlamaları yapınız.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalits',
    r'5[4-5]': 'champions'}


rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

#Görev 5:Aksiyon Zamanı !
#Adım1:Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})

#rfm[["segment","recency", "frequency", "monetary"]].groupby("segment").agg(["mean"])
#Adım2:  RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
#a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#yapankişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

new_df = pd.DataFrame()

new_df = df.merge(rfm, on="master_id", how="left")

#new_df.loc[((new_df["interested_in_categories_12"] == "[KADIN]")

#new_df[new_df["interested_in_categories_12"].str.contains("KADIN", na=False)]


#new_df.loc[((new_df["segment"] == "loyal_customers") | (new_df["segment"] == "champions")) & (new_df["interested_in_categories_12"].str.contains("KADIN", na=False))]



new_woman_shoes_df = pd.DataFrame()

new_woman_shoes_df = new_df.loc[((new_df["segment"] == "loyal_customers") | (new_df["segment"] == "champions"))
                                & (new_df["interested_in_categories_12"].str.contains("KADIN", na=False))]

new_IDs = new_woman_shoes_df["master_id"]

new_IDs = new_IDs.reset_index()

new_IDs.drop("index", axis=1, inplace=True)


new_IDs.to_csv("new_IDs.csv")

#b.Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen
#geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,uykuda olanlar
#ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına
#kaydediniz.

new_discount_df = new_df[(new_df["segment"] == "cant_loose") | (new_df["segment"] == "about_to_sleep") | (new_df["segment"] == "new_customers")]

new_discount_df = new_discount_df[(new_discount_df["interested_in_categories_12"].str.contains("ERKEK","COCUK",na=False))]

new_discount_df = new_discount_df["master_id"]

new_discount_df = new_discount_df.reset_index()

new_discount_df.drop("index", axis=1, inplace=True)

new_discount_df.to_csv("new_discount.csv", index=False)


#### BG-NBD ve Gamma-Gamma ile CLTV Tahmini

#FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir. Şirketin orta uzun vadeli plan yapabilmesi
#için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

#Görev1:  Veriyi Hazırlama
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format',lambda x: '%4f' % x)
from sklearn.preprocessing import MinMaxScaler

#Adım1:flo_data_20K.csv verisini okuyunuz.

df_ = pd.read_csv("Functions_conditions_loop_comprehensions/Projects_doc/flo_data_20k.csv")

df = df_.copy()

df.describe().T

#Adım2:Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını
#tanımlayınız.Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini
#round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

#Adım3:"order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
#"customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

columns=["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df,col)

df.describe().T
df_.describe().T
#for column in df.columns:
#    if 'ever' in column:
#         replace_with_thresholds(df,column)


df.describe().T
#Adım4:Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
#Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

df["custumer_value_total_omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

#Adım5:Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.dtypes

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

df.info()

#Görev 2:CLTV Veri Yapısının Oluşturulması

#Adım1:Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olara kalınız.

df["last_order_date"].max()
#today_date = df["last_order_date"].max() + pd.Timedelta(day=2)
today_date = dt.datetime(2021, 6, 1)

#Adım2:customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv
#dataframe'i oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency vetenure değerleri ise
#haftalık cinsten ifade edilecek.

#recency: Son satin alma uzerinden gecen zaman. Haftalik. (kullanici ozelinde)
#T: Musterinin yasi. Haftalik. (analiz tarihinden ne kadar sure once ilk satin alma yapilmis)
#frequency: tekrar eden toplam satin alma sayisi (frequencty > 1)
#monetary_value: satin alma basina ortalama kazanc (burada gormeye alisik oldugumuz degerler oncekileriyle ayni anlamda degil.)

df.dtypes
cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days / 7  #.astype('timedelta64[D]') calismadi.
cltv_df["T_weekly"] =(today_date - df["first_order_date"]).dt.days / 7
cltv_df["frequency"] = df["order_num_total_omnichannel"]
cltv_df["monetary_cltv_avg"] = df["custumer_value_total_omnichannel"] / df["order_num_total_omnichannel"]

cltv_df.head()

#Görev3:BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
#Adım1:BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

#3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine
#ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,
                                                                                        cltv_df["frequency"],
                                                                                        cltv_df["recency_cltv_weekly"],
                                                                                        cltv_df["T_weekly"])

#6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine
#ekleyiniz.

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                                                        cltv_df["frequency"],
                                                                                        cltv_df["recency_cltv_weekly"],
                                                                                        cltv_df["T_weekly"])

#Adım2:Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyi pexp_average_value
#olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])

cltv_df["pexp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

#Adım3:6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_avg"],
                                              time = 6,
                                              freq = "W")                                 #T'nin frekansi haftalik

#Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv", ascending=False).head(20)

#Görev 4:CLTV Değerine Göre Segmentlerin Oluşturulması
#Adım1:6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba(segmente) ayırınız ve grupisimlerini veri setine ekleyiniz.

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])

cltv_df.groupby("segment").agg({"cltv":["count", "mean", "sum"]})


