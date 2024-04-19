#Data Structures (Veri Yapilari)

#Veri Yapılarına Giriş ve Hızlı Özet

#Sayilar: integer
x = 46
type(x)

#Sayilar: float
x = 10.3
type(x)

#Sayilar: complex
x = 2j + 1
type(x)

#String
x = "Hello ai era"
type(x)

#Boolean
True
False
type(True)

5 == 4

2 == 3

1 == 1

type(3 == 2)

#List

x = ["btc", "eth", "xrp"]
type(x)

#Dictionary (Sozluk)
x = {"name": "Peter", "Age":36 }
type(x)

#Tuple
x = ("python", "ml", "ds")
type(x)

#Set
x = {"python", "ml", "ds"}
type(x)

#NOT:Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections(Arrays) olarak geçmektedir.

### NUMBERS (Sayilar): int, float, complex

a = 5
b = 10.5

a * 3

a / 7

a * b / 10

a ** 2


#Tipleri degistirme
int(b)
float(a)

int(a * b / 10)

c = a * b / 10
int(c)

### STRINGS (Karakter dizileri)

print("John")
print('John')

"John"
name = "John"
name

#Cok Satirli Karakter Dizileri

"""Veri Yapilari: Hizli Ozet,
 Satilar (Numbers): int, float, complex,
 Karakter Dizileri (Strngs): str,
 List, Dictionary, Tuple, Set,
 Boolean (TRUE-FALSE): bool"""

long_str = """Veri Yapilari: Hizli Ozet,
 Satilar (Numbers): int, float, complex,
 Karakter Dizileri (Strngs): str,
 List, Dictionary, Tuple, Set,
 Boolean (TRUE-FALSE): bool"""
long_str

#Karakter Dizilerinin Elemanlarina Erismek

name[0]
name[3]

#Karakter Dizilerinde Slice İslemi

name[0:2]

long_str[0:10]

#String Icerisinde Karakter Sorgulamak

long_str
"veri" in long_str
"Veri" in long_str
"bool" in long_str

#\n alt satira gectigini gosterir.

### String (Karakter Dizisi) Metodlari

dir(int)
dir(str)

#len
name
type(name)
type(len)

len(name)
len("vahitkeskin")
len("miuul")

#upper & lower(kucuk-buyuk donusumler)
"miuul".upper()
"MIUUL".lower()


#Replace(Karakter Degisimi)
hi = "Hello AI Era"
hi.replace("l", "p")

#Split(Bolme)

hi.split()

#Strip(Kirpma)

" ofofo ".strip()
"ofofo".strip("o")

#Capitalize(Ilk harfi buyutur)

"foo".capitalize()

dir(str)
"foo".startswith()
"foo".startswith("f")

### List

# - Degistirilebilirdir.
# - Siralidir. Index islemleri yapilabilir.
# - Kapsayicidir.

notes = [1,2,3,4]
type(notes)

names = ["a", "b", "v", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]
not_nam[0]
not_nam[5]
not_nam[6]
not_nam[6][1]
type(not_nam[6][1])
type(not_nam[6])
notes[0] = 99
notes

not_nam[0:4]

#List Methods(Liste Metodlari)

dir(notes)

len(notes)
len(not_nam)

#append(eleman ekleme)

notes
notes.append(100)
notes

#pop(indexe gore eleman siler)

notes.pop(0)
notes

#insert(indexe ekler)

notes.insert(2, 99)
notes

not_nam.count(1)

not_nam.extend("2")
not_nam.extend([2,3,4])
not_nam.append([2,3,4])

### Dictonary(Sozluk)

# - Degistirelebilirdir.
# - Sirasizdir. (3.7 sonra sirali)
# - Kapsayicidir.

#Key-value

dictionary = {"REG" : "Regression",
              "LOG" : "Logistic Regression",
              "CART" : "Classification and Reg"}
dictionary["REG"]

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}
dictionary["REG"]

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

dictionary["REG"]

dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}
dictionary["CART"][1]

#Key Sorgulama

"REG" in dictionary
"YSA" in dictionary

#Key'e gore Value'ya erismek

dictionary["REG"]
dictionary.get("REG")

#Value Degistirmek

dictionary["REG"] = ["YSA", 10]
dictionary

#Tum Key'lere erismek

dictionary.keys()

#Tum Value'lara Erismek
dictionary.values()

#Tum ciftleri Tuple Halinde Listeye Cevirme
dictionary.items()

#Key-Value degerlerini guncellemek

dictionary.update({"REG": 11})
dictionary

#Yeni Key-Value Eklemek

dictionary.update({"RF": 10})
dictionary

###Tuple(Demet)

# - Degistirilemez.
# - Siralidir.
# - Kapsayicidir.

t = ("john", "mark", 1, 2)
type(t)
t[0]
t[0:3]
t[0] = 99

t = list(t)
t[0] = 99
t = tuple(t)
t


###Set

# - Degistirilebilir.
# - Sirasiz ve essizdir.
# - Kapsayicidir.

#Difference() Iki kumenin farki

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

#Set1 de olup set2 de olmayanlar
set1.difference(set2)
set1 - set2

#Set2 de olup set1 de olmayanlar
set2.difference(set1)
set2 - set1

#symmetric_difference(): Iki kumede birbirlerine gore olmayanlar
set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

#intersection(): Iki kumenin kesisimi

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])
set1.intersection(set2)
set2.intersection(set1)

set1 & set2

#union(): iki kumenin birlesimi

set1.union(set2)

#isdisjoint(): iki kumenin kesimi bos mu?

set1 = set([7, 8, 9])
set2= set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

#issubset(): Bir kume diger kumenin alt kumesi mi?
set1.issubset(set2)
set2.issubset(set1)

#issuperset(): Bir kume diger kumeyi kapsiyor mu?

set2.issuperset(set1)
set1.issuperset(set2)

name = "John"

name[:2]


a = "ali ceylan"
c = a.join("b")
c


Input: ["1, 3, 4, 7, 13", "1, 2, 4, 13, 15"]
#Output: 1,4,13
A = []
B = []
def intersections(A,B):
    if A.intersection(B):
        print(A.intersection(B))


i = 1
    while i < 6:
        print(i)
        i += 1
    else:
        print("i is no longer than 6")

def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))
class MyClass:
  x = 5

def number_check(number):
    if number == 10:
        print("The", number, "is 10")

number_check(10)

#1'den 1000'e kadar olan sayılardan mükemmel sayı olanları ekrana yazdırın. Bunun için bir sayının mükemmel olup olmadığını dönen bir tane fonksiyon yazın.

def perfect_number(number):

    total = 0

    for i in range (1,number):
        if number % i == 0:
            total += i

    return total == number

for i in range(1,1000):
    if(perfect_number(i)):
        print(i, "is perfect number")


salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

double_salary = [salary * 2 for salary in salaries]

[salary * 2 for salary in salaries  if salary < 3000]


[salary * 2 if salary < 3000 else salary * 1.8 for salary in salaries]


students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]


[student.upper() if student not in students_no else student.lower() for student in students]

dictionary = {"a" : 1,
              "b" : 2,
              "c" : 3,
              "d" : 4}

{k : v ** 2 for (k,v) in dictionary.items()}

{k.upper(): v for (k,v) in dictionary.items()}

{k.upper(): v * 2 for (k,v) in dictionary.items()}

#Amac: cift sayilarin karesi alinarak bir sozluge eklenmek istemektedir.
# Key'ler orijinal degerler value'lar ise degistirilmis degerler olacak.

numbers = range(10)

new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

{n : n ** 2 for n in numbers if n % 2 == 0}


### Conditions(Kosullar)
from typing import Callable, Any

import pandas as pd

1 == 1
1 == 2

# If

if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11
if number == 10:
    print("number is 10")

number = 10


def number_check(number):
    if number == 10:
        print("number is 10")


number_check(12)

number_check(10)


# Else

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")


number_check(12)


# Elif

def number_check(number):
    if number > 10:
        print("Greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")


number_check(10)

number_check(18)

number_check(6)

###Loops(Donguler)

# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary * 20 / 100 + salary))

for salary in salaries:
    print(int(salary * 30 / 100 + salary))

for salary in salaries:
    print(int(salary * 50 / 100 + salary))


def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)


new_salary(1500, 10)
new_salary(2000, 20)

for salary in salaries:
    print(new_salary(salary, 10))

salaries2 = [10700, 25000, 30400, 40300, 50200]

for salary in salaries2:
    print(new_salary(salary, 15))

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))

# Uygulama - Mulakat Sorusu

# Amac: Asagidaki sekilde string degistiren fonksiyon yazmak istiyoruz.

# Before: "hi my name is john and i am learning python"
# After: "Hi mY NaMe  iS JoHn aNd i aM LeArNiNg pYtHon"

range(len("miuul"))
range(0, 5)

for i in range(0, 5):
    print(i)

for i in range(len("miuul")):
    print(i)

4 % 2 == 0
m = "miull"
m[0]


def alternating(string):
    new_string = ""
    # girilen string'in index'lerinde gez.
    for string_index in range(len(string)):
        # index cift ise buyuk harflere cevir
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise kucuk harflere cevir
        else:
            new_string += string[string_index].lower()
    print(new_string)


alternating("miull")

alternating("hi my name is john and i am learning python")

# Break & Continue & While

salaries = [1000, 2000, 3000, 4000, 5000]

# Break

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

# Continue

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

# While

number = 1
while number < 5:
    print(number)
    number += 1

# Enumerat: Otomatik counter/Indexer ile for loop


students = ["John", "Mark", "Venessa", "Mariam"]

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)

A
B
# Uygulama - Mülakat Sorusu

# divide_students fonksiyonunu yazınız.
# Cift indexte yer alan ogrencileri bir listeye aliniz.
# Tek indexte yer alan ogrencileri baska bir listeye aliniz.
# Fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]


def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups


st = divide_students(students)

st[0]

st[1]


# alternating fonksiyonunun enumerate ile yazislmasi

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumerate("hi my name is john and i am learning python")

# Zip

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathematics", "statistics", "physics", "astronomy"]

ages = ["23", "30", "26", "22"]

list(zip(students, departments, ages))


# Lambda, map, filter, reduce

# Lambda

def summer(a, b):
    return a + b


summer(1, 3) * 9

new_sum = lambda a, b: a + b

new_sum(4, 5)

# Map

salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


new_salary(5000)

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

# del new_sum

list(map(lambda x: x * 20 / 100 + x, salaries))

list(map(lambda x: x ** 2, salaries))

# Filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, list_store))

# Reduce
from functools import reduce

list_store = [1, 2, 3, 4]
reduce(lambda a, b: a + b, list_store)


###Compregensions

#List Comprehension

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary((salary)))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary*2))


[salary* 2 for salary in salaries]


[salary* 2 for salary in salaries if salary < 3000]


[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]


[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]


students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in students_no else student.lower() for student in students]

#Dict Comprehension

dictionary = {"a":1,
              "b":2,
              "c":3,
              "d":4}

dictionary.keys()
dictionary.values()
dictionary.items()


{k: v ** 2 for (k, v) in dictionary.items()}


{k.upper(): v for (k, v) in dictionary.items()}


{k.upper(): v * 2 for (k, v) in dictionary.items()}


#Uygulama-Mulakat Sorusu

#Amac: cift sayilarin karesi alinarak bir sozluge eklenmek istemektedir.
#Key'ler original degerler value'lar ise degistirilmis degelerler olacaktir.


numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

new_dict

{n : n ** 2 for n in numbers if n % 2 == 0}

# List & Dict Comprehension Uygulamalari

# Bir Veri Setindeki Degisken İsimlerini Degistirmek

# before:
# ["total", "speeding", "alcohol", "not_distracted", "no_proveious", "ins_premium", "ins_losses", "abbrev"]

# after:
# ["TOTAL", "SPEEDING", "ALCOHOL", "NOT_DISTRACTED", "NO_PREVİOUS", "INS_PREMIUM", "INS_LOSSES", "ABBREV"]


import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col)

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df.columns = [col.upper() for col in df.columns]

df.columns

#Isminde "INS" olan degiskenlerin basina "FLAG" digerlerine "NO_FLAG" eklemek istiyoruz.

# before:
# ["TOTAL", "SPEEDING", "ALCOHOL", "NOT_DISTRACTED", "NO_PREVİOUS", "INS_PREMIUM", "INS_LOSSES", "ABBREV"]


# after:
# ["NO_FLAG_TOTAL", "NO_FLAG_SPEEDING", "NO_FLAG_ALCOHOL", "NO_FLAG_NOT_DISTRACTED", "NO_FLAG_NO_PREVİOUS", "FLAG_INS_PREMIUM", "FLAG_INS_LOSSES", "NO_FLAG_ABBREV"]

[col for col in df.columns if "INS" in col]

["FLAG_" + col for col in df.columns if "INS" in col]


["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

df.columns


#Amac key'i string, value'su asagidaki gibi bir liste olan sozluk olusturmak ve sadece sayisal degiskenler icin yapmak istiyoruz.

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

#object olmayan degiskenleri getirdik.
num_cols = [col for col in df.columns if df[col].dtype != "O"]

soz = {}
agg_list = ["mean", "min", "max", "var"]

for col in num_cols:
    soz[col] = agg_list

soz

#Kisa yol:
{col : agg_list for col in num_cols}


new_dict = {col : agg_list for col in num_cols}

df[num_cols].head()

df[num_cols].agg(new_dict)

liste = [1, 2, 3, 4, 5]
yeni_liste = [i for i in liste]
print(yeni_liste)

eski_fiyat = {'süt': 1.02, 'kahve': 2.5, 'ekmek': 2.5}

dolar_tl = 0.76
yeni_fiyat = {item: value*dolar_tl for (item, value) in eski_fiyat.items()}
print(yeni_fiyat)

def interrogator(number):
    if number % 2 == 0:
        print("Cift sayidir")
    else:
        print("Sayi tek sayidir")

interrogator(8)

def interrogator2(number):
    if number >= 0:
        print("Pozitif sayidir.")
    else:
        print("Negatif sayidir")

interrogator2(0.1)

def distance(number):
    if number < 0:
        print(float(-number))
    else:
        print(float(number))

distance(-3)

def small_number(number1, number2):
    if number1 < number2 :
        print(number1, "is smaller than", number2)
    elif number2 < number1:
        print(number2, "is smaller than", number1)
    else:
        number1 = number2
        print(number1, "and", number2, "are equal")

small_number(1, 1)

small_number(4, 35)

def average_numbers(number1, number2):

for i in range(1,101):
    if 0 == i % 3 and i % 5 == 0:
    print(i, "15'in katlaridir.")


def diktorgen_cevre_hesabi(kisa_kenar, uzun_kenar):
    print((kisa_kenar + uzun_kenar) * 2)

diktorgen_cevre_hesabi(4,5)

def ciflerin_ortalamasi(ilk_sayi, ikinci_sayi):

    groups = [[], []]
    for sayi in ciflerin_ortalamasi(ilk_sayi,ikinci_sayi):
        if(sayi % 2 == 0):
            groups[0].append(sayi)
        else:
            groups[1].append(sayi)
    print(groups)
    return(groups)

ciflerin_ortalamasi(1,25)


def myfunc():
    global x
    x ="fantastic"

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
    print(col.upper())

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")
df.columns =[col.upper() for col in df.columns]

["FLAG_" + col for col in df.columns if "INS" in col]

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]


### Data Analysis With Python

# - Numpy
# - Pandas
# - Veri Görsellestirme: Matplotlib & Seaborn
# - Advanced Functional Exploratory Data Analysis(Gelismis Fonksiyonel Kesifci Veri Analizi)

#Neden Numpy?
import numpy as np
a= [1, 2, 3, 4]
b= [2, 3, 4, 5]

ab = []

for i in range(0, len(a)):
    ab.append(a[i]*b[i])


a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

a*b

# - Numpy listelere gore daha hizlidir. Sabit tipte veri tutar yani verimli veri saklar bundan dolayi hizlidir.
# - Yuksek seviyeden islemler yapma imkani saglar.

#Creating NumPy Arrays(NumPy Array'i olusturmak)

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.zeros(10, dtype=int)
np.random.random_integers(0, 10, size=10)
np.random.randint(0, 10, 10)
np.random.normal(10, 4, (3,4))

#Attibutes of NumPy Arrays(NumPy Array Ozellikleri)

#ndim: boyut sayisi
#shape: boyut bilgisi
#size: toplam eleman sayisi
#dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

#Reshaping (Yeniden Sekillendirme)

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3, 3)

ar = np.random.randint(1, 10, size=9)
ar.reshape(3,3)

#Index Selection(Index Secimi)

a = np.random.randint(10, size=10)
a[0]
a[0:5] #slicing
a[::-1]
a[0] = 999
a
m = np.random.randint(10, size=(3,5))
m[2,4]
m[1,1]
m[2,3] = 999
m.dtype
m[2,3] = 2.9
#Numpy fix type arraydir sabit tipli arraydir. Verimli veri saklama yönünden.
m[:, 0]
m[1, :]
m[0:2, 0:3]

#Fancy Index

v = np.arange(0, 30, 3)
v[1]
v[4]
#numpy arrayine  bir liste girdigimizde(indeks numarası ya da true,false ifadeler) secim islemi saglar.
catch = [1, 2, 3]

v[catch]

#Conditions on NumPy(NumPy'da kossullu islemler)

a = [1, 2, 3, 4, 5]

#-Klasik dongu ile
ab = []

for i in a:
    if i < 3:
       ab.append(i)

#- Numpy ile
import numpy as np
v = np.array([1, 2, 3, 4, 5])

v < 3

v[v < 3]
v[v > 3]
v[v != 3]
v[v == 3]
v[v >= 3]
v[v <= 3]

# Mathematical Operations(Matematiksel Islemler)

v = np.array([1, 2, 3, 4, 5])

v / 5
v * 5 / 10
v ** 2
v - 1
np.subtract(v, 1)  #cikarma islemi
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

#NumPy ile Iki bilinmeyenli Denklem Cozumu

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = ([[5, 1], [1, 3]])
b = ([12,10])
np.linalg.solve(a, b)

### Pandas

#veri manupulasyonu ya da veri analizi dendiginde akla gelen ilk python kutuphanelerinden biridir.

#Pandas Series
import pandas as pd

s = pd.Series([10, 77, 12, 4, 5]) #pandas serisi olusturma.
type(s)
s.index
s.dtype
s.size
s.ndim #boyut bilgisi
#Bir pandas serisinin sonuna values ifadesini girdiğimizde ve degerlerine erismek istedigimiz bu durumda indeks bilgisi
#ile ilgilenmiyoruz demis oldugumuzdan dolayi bize bunu bir numpyarray'e dondurdu.
s.values
type(s.values)
s.head(3)
s.tail(3)

#Reading Data(Veri Okuma)
import pandas as pd

df = pd.read_csv(".idea/machine-readable-business-employment-data-sep-2023-quarter.csv")
df.head()

#Quick Look at Data(Veriye Hizli Bakis)

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull()
df.isnull().values
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

# Selection in Pandas(Pandas Secim Islemleri)
import pandas as pd
import  seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.index
df[0:7]
df.drop(0 ,axis=0).head()               #(axis=0 satir, axis=1 sutun)
delete_indexes = [1 ,3 ,5 ,7]
df.drop(delete_indexes, axis=0).head(10)
# df = df.drop(delete_indexes, axis=0)           kalıcı olarak silmek icin tekrar df olarak atariz.
# df.drop(delete_indexes, axis=0, inplace=True)  kalıcı olarak silmek icin. inplace kalicilik saglar.
# Degiskeni indexe cevirme
df["age"].head()
df.age.head()
df.index
df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)
#index'i degiskene cevirmek istersek
df.index
#df'in icine girecek olacagimiz string ifade dataframein icinde varsa bu durumda bu degisken secilir. Eger yoksa
#yeni degisken eklendigi anlasilir.
df["age"] = df.indexdf.head()
df.drop("age", axis=1, inplace=True)
df = df.reset_index() #indexte yer alan degeri siler ardindan bunu sutun olarak yeni bir degisken olarak ekler.
df.head()

#Degiskenler Uzerinde Islemler
import pandas as pd
pd.set_option("display.max_columns", None) #gosterilecek olan max. kolon sayisi olmasin. Hepsini goster.
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()
df.age

df[["age"]] #Bir degisken secerken sonucu seri(pandas) ya da dataframe olarak alaiblirsiniz. İki koseli parantez ile df olmaya devam eder.
type(df["age"].head())
type(df[["age"]].head())

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

type(df[col_names])

df["age2"] = df["age"]**2
df.head()
df["age3"] = df["age"]/ df["age2"]
df.drop("age3", axis=1).head()
df.drop(col_names , axis=1).head()
df.loc[:, ~df.columns.str.contains("age")].head() #(~(tilda) disindakiler demektir. Burada da
                                                  # ifadenin disindakileri sec dedik.

#iloc & loc
#DataFramelerde secim islemleri icin kullanilan ozel yapilardir. iloc Numpy'dan listelerden alışık olduğumuz klasik
#int based yani index bilgisi vererek secim yapma islemlerini ifade eder. Acilimi da integer based selectiondur.
#loc ise mutlak olarak indekslerdeki labellara gore secim yapar. label based selectiondur. loc ne yaziyorsa onu
#getirir.

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.iloc[0:3]
df.iloc[0, 0]

df.loc[0:3]

df.iloc[0:3, 0:3]
df.iloc[0:3, "age"] #calismaz indexlere gore islem gorur.
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

#Conditional Selection (Kosullu Secim)

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50].count() #butun degiskenlere count atar.
df[df["age"] > 50]["age"].count()

df.loc[df["age"]>50, "class"].head() #Yasi 50den buyuk olan kisilerin sinif bilgisi.

df.loc[df["age"]>50, ["age", "class"]].head() #yas bilgisini de bir liste ile getirdik.

df.loc[(df["age"]>50) & (df["sex"]== "male"), ["age", "class"]].head() #Birden fazla kosul varsa parantez icine almaliyiz.
                                                                       # ve "&" isareti ile birlestirebiliriz.
df.head()

df.loc[(df["age"]>50) & (df["sex"]== "male") & (df["embark_town"]=="Cherbourg"), ["age", "class", "embark_town"]].head()

df["embark_town"].value_counts()

df.loc[(df["age"]>50)
       & (df["sex"]== "male")
       & ((df["embark_town"]=="Cherbourg") | (df["embark_town"]=="Southampton")),  # | ya da anlamına gelir.
       ["age", "class", "embark_town"]].head()

#Aggregation & Grouping(Toplulastirma ve Gruplama)
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

df.groupby("sex")["age"].mean()   #groupby = kirilimi ifade eder. Gruplara ayirmayi saglar gibi dusunebiliriz.

df.groupby("sex").agg({"age": "mean"})              #bu kullanim birden cok fonksiyonu uygulamamizi saglar.
df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": ["mean"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                                                "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({
    "age": ["mean"],
    "survived": "mean",
    "sex": "count"})

#Pivot Table

df.pivot_table("survived", "sex", "embarked") #Pivot_tablein on tanimli degeri meandir.
df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", "embarked", aggfunc=["std","sum"])


#birden fazla durum olacaksa liste yontemiyle yapariz.
df.pivot_table("survived", "sex", ["embarked", "class"])

#sayisal bir degiskeni kategorik degiskene cevirerek pivot_table a ekledik.(cut ve qcut fonksiyonlarini kullaniriz.
#sayisal degiskeni neye bolecegimizi biliyorsak cut fonksiyonu, ceyreklik degerlere bolunsun istiyorsak qcut fonksiyonu
#kullanilir.)
df["new_age"] = pd.cut(df["age"],[0, 10, 18, 25, 40, 90])
df.head()

df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex",[ "new_age","class"])

#pd.set_option('display.witdh', 500) yanyana cikti almamizi saglar.


df.pivot_table("survived", "sex", "new_age")

#Apply ve Lambda
import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
#apply:satır ya da sutunlarda otomatik calisabilen yapilardir. Satır ya da sutunlarda otomatik olarak fonksiyon
#calistirmaya saglar.
#lambda:fonksiyon tanimlamadan kullan at fonksiyonlar tanimlama imkani saglar.

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10

df[['age', 'age2', 'age3']].apply(lambda x : x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x : (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.head()

#Join (Birlestirme)
import pandas as pd
import numpy as np
m = np.random.randint(1, 30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2])  #dataframeleri birlestirmeye yarar. axis=1 yaparsak yanyana birletirir.
pd.concat([df1, df2], ignore_index=True) #indeks bilgilerini duzeltmemize yarar.

#Merge ile Birlestirme islemleri

df1 = pd.DataFrame({'employees': ['john', 'dennis', 'mark', 'maria'],
                    'group': ['accounting', 'engineering', 'engineering', 'hr']})

df2 = pd.DataFrame({'employees': ['mark', 'john', 'dennis', 'maria'],
                    'start_date': [2010, 2009, 2014, 2019]})

pd.merge(df1, df2)

pd.merge(df1, df2, on="employees") #ozellikle hangi degiskene gore birlesitirmek istiyorsak on kullaniriz.

#Amac: her calisanin mudurunun bilgisine erismek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({'group': ['accounting', 'engineering', 'hr'],
                    'manager': ['Caner', 'Mustafa', 'Berk']})

pd.merge(df3, df4)

###Veri Görsellestirme

#Matplotlib: Pythondaki veri gorsellestirmenin atasidir. Low leveldır.

#Kategorik degisken: Sutun Grafik. barplot(matplotlib), countplot(seaborn)
#Sayisal degisken: hist, boxplot

#Kategorik Degisken Gorsellestirme

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
from matplotlib import pyplot as plt
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()

#Sayisal Degisken Gorsellestirme

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])

plt.boxplot(df["fare"])

#Matplotlib Ozellikleri
#Yapisi itibari ile katmanli sekilde veri gorsellestirmeyi saglar.


import pandas as pd
import numpy as np
import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#plot: veriyi gorsellestirmek icin kullandigimiz fonksiyonlardan bir tanesi.

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.plot(x, y, 'o') #belirtilen yerlere nokta koyar. Eger grafik kapanmadan girdi olursa var olan grafigin uzerine cizim yapar.

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)

plt.plot(x, y, 'o')

#Marker

y= np.array([13, 28, 11, 100])

plt.plot(y, marker='o')

plt.plot(y, marker='+')

markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']

#line

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashed")

plt.plot(y, linestyle="dashdot")

plt.plot(y, linestyle="dashdot", color="r")

#Multiple lines

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#Labels

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
#baslik
plt.title("Bu ana başlık")

#X eksenini isimlendirme
plt.xlabel("X eksenini isimlendirme")

#Y eksenini isimlendirme
plt.ylabel("Y eksenini isimlendirme")

plt.grid() #arka tarafa izgara ekler.
plt.show()

#Subplots

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 1) #1 satırlık 2 sutunluk bir grafigin 1. grafigini olusturmak istiyorum demek.
plt.title("1")
plt.plot(x, y)

x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 20, 26, 27, 280, 29, 30, 30, 30, 30])
plt.subplot(1, 2, 2) #1 satırlık 2 sutunluk bir grafigin 2. grafigini olusturmak istiyorum demek.
plt.title("2")
plt.plot(x, y)

#Seaborn

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df['sex'].value_counts().plot(kind='bar') #matplot
plt.show

#Sayisal degisken gorsellestirme

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

###Advanced Functional Eda(Gelismis Fonksiyonel Kesifci Veri Analizi)

#Hizli bir sekilde genel fonksiyonlar ile elimize gelen verileri analiz etmek


#Genel Resim

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print("############## Shape #############")
    print(dataframe.shape)
    print("############## Types #############")
    print(dataframe.dtypes)
    print("############## Head #############")
    print(dataframe.head(head))
    print("############## Tail #############")
    print(dataframe.tail(head))
    print("############## NA #############")
    print(dataframe.isnull().sum())
    print("############## Quantiles #############")
    print(dataframe.describe([0, 0.005, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df = sns.load_dataset("tips")
check_df(df)

df = sns.load_dataset("flights")
check_df(df)

#Analysis of Categorical Variables (Kategorik Degisken Analizi)

#cok fazla degisken oldugunda bunu tek tek yakalayamayacagimizdan dolayi onemli.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()
df.info()

df["sex"].dtypes
#dtype('O') ciktisini verir.
str(df["sex"].dtypes)
#object ciktisini verir

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
#kategorik olup sayisal degisken gibi olan degiskenleri de ayiklamaliyiz. "survived" gibi.

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

cat_but_car = [col for col in df.columns if df[col].nunique() >20 and str(df[col].dtypes) in ["object", "category"]]
#olculebilir olmayan kategorik degiskenleri ariyoruz.

cat_cols = cat_cols + num_but_cat

cat_cols =  [col for col in cat_cols if col not in cat_but_car] #cat_but_car bos kume olmasaydi bu islemi uygulamaliydik.

df[cat_cols]

df[cat_cols].nunique()
#tutarli olup olmadigini kontrol ediyoruz.

[col for col in df.columns if col not in cat_cols] #sayisal degiskenlerin secimi


def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################################################################")
cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe,col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex")
cat_summary(df, "sex", plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

#Alternatif asagidaki bool hatasi icin
for col in cat_cols:
    if df[col].dtypes == "bool":
        continue
    cat_summary(df, col, plot=True)



#Adault Male Bool oldugu icin hata aliriz. Bu hatayi almamak icin
for col in cat_cols:
    if df[col].dtypes == "bool":
        print("do not come graphic")
    else:
    cat_summary(df, col, plot=True)

#bool tipi degistirmek istersek;
df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

#Do one thing. Dongu disarida yazilmalidir.

#yukaridaki if else yapisini fonksiyonun icinde yapalim bicimlendirmeye calisalim. Ancak tercih etmeyiz pek.
def cat_summary(dataframe,col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)


        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),

                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#######################################################################")

        if plot:
             sns.countplot(x=dataframe[col_name], data=dataframe)
             plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),

                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#######################################################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "sex")

cat_summary(df, "sex", plot=True)

cat_summary(df, "adult_male", plot=True)


#Analysis of Numerical Variables (Sayisal Degisken Analizi)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
cat_but_car = [col for col in df.columns if df[col].nunique() >20 and str(df[col].dtypes) in ["object", "category"]]
cat_cols = cat_cols + num_but_cat
cat_cols =  [col for col in cat_cols if col not in cat_but_car]

df[["age","fare"]].describe().T

#Sayisal degiskenleri sectik
num_cols = [col for col in df.columns if df[col].dtypes in ["int64","float64"]]

#Ancak numerik gibi gozukup kategorik olan degerleri de ayiklamaliyiz.
num_cols = [col for col in df.columns if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] #ne alaka?
    print(dataframe[numerical_col].describe(quantiles).T)
# veri setimizi küçükten büyüye sıraladığımızda en kücükten en büyüğe doğru ilerlerken %5’lik
#dilimdeki değer 4 demek istiyor(quantiles)


num_summary(df, "age")

for col in num_cols:
    num_summary(df,col)

def num_summary(dataframe, numerical_col, plot=False):
    qunatiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] #ne alaka?
    print(dataframe[numerical_col].describe(qunatiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#Capturing Variables and Generalizing Operations(Degiskenlerin Yakalanmasi ve Islemlerin Genellestirilmesi)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()
 #bir degisken sayisal olsa dahi essiz sinif sayisi(car_th) 10dan kucukse bu bir kategorik degiskendir muammelesi
 #yapacagiz. Yine bir kategorik degisken essiz deger sayisi 20den buyukse buna kardinal degisken muamelesi yapacagiz.

#docstring(arama bolumune bunu yazip numpya donusturebilirsin.) fonksiyora dokuman yazmak.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe:dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th:int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal degişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişen listesi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un icerisinde.


    """


    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["object", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in dataframe.columns if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_car: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

grab_col_names(df)

def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################################################################")
cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    qunatiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] #ne alaka?
    print(dataframe[numerical_col].describe(qunatiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#BONUS
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe,col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

#Analysis of Target Variable(Hedef Degisken Analizi)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe,col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)


        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),

                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#######################################################################")

        if plot:
             sns.countplot(x=dataframe[col_name], data=dataframe)
             plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),

                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("#######################################################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe:dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th:int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal degişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişen listesi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un icerisinde.


    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["object", "category"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in dataframe.columns if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_car: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#returunden dolayi tutmak icin grabe esitliyoruz.
df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

#Hedef Degiskenin Kategorik Degiskenler ile Analizi

df.groupby("sex")["survived"].mean()
#Hedef degiskenin diger degiskenlerle alakasini arastiriyoruz.

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived",col)

#Hedef Degiskenin Sayisal Degiskenler ile Analizi

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

#Analysis of Correlation(Korelasyon Analizi)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, 2:-1] #istenmeyen degiskenlerden kurtuluyoruz ancak anlamadim??
df.head()

df.info()
#Not: Yuksek korelasyonlu degiskenlerden birini her calismada silmek zorunda degiliz. Sadece ihtiyacimiz oldugunda
#kullaniriz. Ihtiyacimiz oldugunda sadece bir analiz araci olarak kullanmaliyiz. Korelasyonda 1e yaklastikca iliski
#siddeti kuvvetli -1e yaklastikca da negatif yonlu iliski kuvvetlidir.

#numerik degiskenleri sececek fonksiyonumuzu yazalim.
num_cols = [col for col in df.columns if df[col].dtype in ["int64","float64"]]
#korelasyonlari hesaplama icin corr() kullaniyoruz.
corr = df[num_cols].corr()

#isi haritasi olusturuyoruz.
sns.set(rc={'figure.figsize': (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


#Yuskek Korelasyonlu Degiskenlerin Silinmesi

#(kosegenler ve altinda kalan alan silinecek)
cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as pls
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)

### CRM ANALYTCS ###

###Introduction to CRM Analytics(CRM Analitigine Giris)(CRM = Customer Relationship Management = Musteri İliskileri Yonetimi)

#-CRM:Customer Relationship Management(Musteri İliskileri Yonetimi)
#-Musteri yasam dongusu optimizasyonlari(customer lifecycle/ journey/ funnel)
#-Iletisim(dil, renk, gorseller, kampanyalar)
#-Musteri edinme/bulma calismalari
#-Musteri elde tutma(terk) calismalari
#-Cross-sell(Capraz Satis, tamamlayici urunlerin satisini ifade eder), Up-sell(Ust Satis, daha buyuk boyutta satis)
#-Musteri Segmentasyon Calismalari
#Amacimiz: Tum musteri iliskileri surecini veriye dayali olarak daha verimli hale getirmeye calismaktir.

#KPIs - Key Performance Indicators (Temel Performans Göstergeleri)
#Sirket, departman ya da calisanlarin performanslarini degerlendirmek icin kullanilan matematiksel gostergelerdir.
#Customer Acquisition Rate(Musteri Kazanma Orani)
#Customer Retention Rate(Musteri Elde Tutma Orani)
#Customer Churn Rate(Musteri Terk Orani)
#Conversion Rate(Donusum Orani)
#Growth Rate(Buyume Orani)

#Analysis of Cohort(Kohort Analizi)
#Ortak ozelliklere sahip bir grup insan davranisinin analizidir.

##Customer Segmentation with RFM(RFM ile Musteri Segmantasyonu)(Kural tabanli musteri segmentasyon yontemidir.)
#RFM:Recency, Frequency, Monetary
#RFM Analizi musteri segmentasyonu icin kullanilan bir tekniktir. Musterilerin satin alma aliskanliklari uzerinden
#gruplara ayirilmasini ve bu gruplar ozelinde stratejiler gelistirebilmesini saglar. CRM calismalari icin bircok
#baslikta veriye dayali aksiyon alma imkani saglar.
#RFM Metrikleri: Recency(Yenilik,en son ne zaman alisveris yaptigi gibi) Frequency(Siklik), Moneytary(Parasal deger)
#RFM Skorlari: Metrikleri kiyaslayabilmek icin RFM skorlarina cevirmemiz gerekir. Hepsini kiyaslayabilecegimiz
#formata getirecegiz.(1 ile 5 arasinda, Recencyde puanlama ters orantili olacak sekilde yapilir. RFM degerlerinden
#gelen ifadeler yanyana getirildiginde RFM skorlari olusur.)
#RFM skorlari uzerinde Segment olusturmak(monetary degeri olmaz onun uzerinde yorum yapmak daha az anlamli olur.)

######################################



###Customer Segmantation with RFM(RFM ile Musteri Segmantasyonu)
##1-) İs problemi

#Bir e-ticaret sirketi musterilerini segmentlere ayirip bu segmentlere gore pazarlama stratejilerini belirlemek istiyor.

#Veri Setinin Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli online bir satis magazasinin 01/12/2009 - 09/12/2011 tarihleri
#arasindaki satislarini iceriyor.

#Degiskenler
# InvoiceNo: Fatura numarasi. Her isleme yani faturaya ait essiz numara. C ile basliyorsa iptal edilen islem.
# StockCode: Urun kodu. Her bir urun icin essiz numara.
# Description: Urun ismi
# Quantity: Urun adedi. Faturalardaki urunlerden kacar tane satildigini ifade eder.
# InvoiceDate: Fatura tarihi ve zamani.
# UnitPrice: Urun fiyati (Sterlin)
# CustomerID: Essiz musteri numarasi.
# Country: Musterinin yasadigi ulke.

##2-)Data Understanding (Veriyi Anlama)
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x) #Sayisal degiskenlerin virgulden sonra 3 basamak

df_ = pd.read_excel("Functions_conditions_loop_comprehensions/Projects_doc/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()

#Essiz urun sayisi nedir?
df["Description"].nunique()

df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).head() #bir problem var...

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending = False).head()

df["Invoice"].nunique()

df["TotalPrice"] = df["Quantity"] * df["Price"]

df.head()

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

##3-)Data Preparation(Veri Hazirlama)

df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.shape

df.describe().T

df = df[~df["Invoice"].str.contains("C", na=False)]

##4-) Calculating RFM Metrics(RFM Metriklerinin Hesaplanmasi)

#Recency, Frequency, Monetary
df.head()

df["InvoiceDate"].max()

today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                     "Invoice": lambda num: num.nunique(),
                                     "TotalPrice": lambda price: price.sum()})
rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]
rfm.describe().T #inceledigimizde monetary min degeri sifirdir. Bunu kaldirmaliyiz.
rfm = rfm[rfm["monetary"] > 0]
rfm.describe().T #buldugumuz rfm metriklerini rfm skoruna cevirmeliyiz.

##5-) Calculating RFM Scores(RFM Skorlarinin Hesaplanmasi)

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]) #qcut bir degeri kucukten buyuge siralar
                                                                             #belirli parcalara gore boler.
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

#yigilma olabileceginden dolayi bir degerde o deger diger siniflarin icine tasabilir. Buna engel olmak amaciyla
#rank(method = "first") metodunu kullaniriz.
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm["recency_score"]. astype(str) +
                    rfm["frequency_score"].astype(str))

rfm.describe().T #ekledigimiz skorlar gozukmedi cunku string tipteler.

rfm[rfm["RFM_SCORE"] == "55"]

rfm[rfm["RFM_SCORE"] == "11"]

##6-)Creating & Analysing RFM Segments(RFM Segmentlerinin Olusturulmasi ve Analiz Edilemesi)
#regex=seg_mapin cikti vermesini saglar,matchler aslinda..
#regex=seg_mapin cikti vermesini saglar,matchler aslinda..
#RFM isimlendirmesi
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

rfm["segment"] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "cant_loose"].head()
rfm[rfm["segment"] == "cant_loose"].index #sadece id numaralarina ulasmak istersek

new_df = pd.DataFrame()   #yeni bir df olusturup istenilen grubu icine aktarabiliriz.
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int) #ondalik degerlerden kurtulmak istersek

new_df.to_csv("new_customers.csv") #reload from disk diyerek yeniliyoruz ardindan csv dosyasi geliyor.

##7-) Tum Surecin Fonksiyonlastirilmasi / Surecin Scripte Cevrilmesi

def create_rfm(dataframe, csv=False):


    #Veriyi Hazirlama
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    #RFM Metriklerinin Hesaplanmasi
    today_date = dt.datetime(2010, 12, 11)
    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                                "Invoice": lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[(rfm["monetary"] > 0)]

    #RFM Skorlarinin Hesaplanmasi
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    #cltv_df skorlari kategorik degere donusturulup df'e eklendi
    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                        rfm["frequency_score"].astype(str))


    #Segmentlerin Isimlendirilmesi
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

    rfm["segment"] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["segment", "recency", "frequency", "monetary"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")
    return rfm

rfm_new = create_rfm(df, csv=True)

###Customer Lifetime Value(Musteri Yasam Boyu Degeri)
#Bir musterinin bir sirketle kurdugu iliski iletisim suresince bu sirkete kazandiracagi parasal degerdir.
#CLTV = (Customer Value/Churn Rate) * Profit Margin
#Customer Value = Average Order Value * Purchase Frequency
#Average Order Value = Total Price / Total Transction
#Purchase Frequency = Total Transaction / Total Number of Customers
#Churn Rate = 1 - Repeat Rate (birden fazla alisveris yapiyorsa elde tutulan musteridir.)
#Repeat Rate = Birden fazla alisveris yapan musteri sayisi / tum musteriler
#Profit Margin = Total Price * 0.10

#Sonuc olarak her bir musteri icin hesaplanacak olan CLTV degerlerine gore bir siralama yapildiginda ve CLTV degerlerine
#gore belirli noktalardan bolme islemi yapilarak gruplar olusturuldugunda musterimiz segmentlere ayrilmis olacaktir.

################ Customer Lifetime Value Uygulamasi #####################

#Veri Setinin Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli online bir satis magazasinin 01/12/2009 - 09/12/2011 tarihleri
#arasindaki satislarini iceriyor.

#Degiskenler
# InvoiceNo: Fatura numarasi. Her isleme yani faturaya ait essiz numara. C ile basliyorsa iptal edilen islem.
# StockCode: Urun kodu. Her bir urun icin essiz numara.
# Description: Urun ismi
# Quantity: Urun adedi. Faturalardaki urunlerden kacar tane satildigini ifade eder.
# InvoiceDate: Fatura tarihi ve zamani.
# UnitPrice: Urun fiyati (Sterlin)
# CustomerID: Essiz musteri numarasi.
# Country: Musterinin yasadigi ulke.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df_ = pd.read_excel("Functions_conditions_loop_comprehensions/Projects_doc/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.isnull().sum()

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
df = df[(df['Quantity'] > 0)] #eksi sayida deger olamayacagindan bu db ozelinde filtreliyoruz.
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                        "Quantity": lambda x: x.sum(),
                                        "TotalPrice": lambda x: x.sum()})

cltv_c.head()

cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

##2-)Average Order Value (average_order_value = total_price / total_transaction=

cltv_c.head()

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##3-)Purchase Frequency (total_transaction / total_number_of_customer)

cltv_c.head()

cltv_c.shape[0] #total_number_of_customer

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

##4-)Repeat Rate & Churn Rate(birden fazla alisveris yapan musteri sayisi / tum musteriler)

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate

##5-)Profit Margin (profit_margin = total_price * 0.10)

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

##6-)Customer Value (customer_value = average_order_value * purchase_frequency)

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

##7-)Customer Lifetime Value(CLTV = (customer_value / churn_rate) * profit_margin)

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values("cltv", ascending=False)

##8-)Creating Segments(Segmentlerin Olusturulmasi)

cltv_c.sort_values("cltv", ascending=False).head()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_c.sort_values("cltv", ascending=False).head()

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltv_c.csv")

#Functionalization(Tum Islemlerin Fonksiyonlastirimasi)

def create_cltv_c(dataframe, profit=0.10):

    #Veriyi Hazirlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                                   "Quantity": lambda x: x.sum(),
                                                   "TotalPrice": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    #avg_order_value
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    #repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    #profit margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * profit

    #Customer Value
    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

    #Customer Life Value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

    #Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c

df = df_.copy()

clv = create_cltv_c(df)

###Customer Lifetime Value Prediction (Musteri Yasam Boyu Degeri Tahmini)

#Customer Value = Purchase Frequency * Average Order Value

#CLTV = Expected Number of Transaction * Expected Average Profit

#CLTV = BG/NBD Model * Gamma Gamma Submodel







###BG/NBD(Beta Geometric / Negative Binomial Distribution) ile Expected Number of Transaction
#expected:bir rassal degiskenin beklenen degeri
#rassal degiskenin beklenen degeri o rassal degiskenin ortalamasi demektir.
#Degerlerini bir deneyin sonucundan alan degiskenlere rassal degisken denir.

#BG/NBD: Buy Till You Die (satin alma ve satin almayi birakma surecini olasiksal olarak modeller)

#BG/NBD Expeted Number of Transaction icin iki sureci olasiliksal olarak modeller.
#Transaction Process(Buy) + Dropout Process(Till you die)

#Transaction Process(Buy)
#Alice oldugu surece, belirli bir zaman periyodunda, bir musteri tarafindan gerceklestirilecek islem sayisi transaction
#rate parametresi ile Possion ddagilir.
#Bir musteri alive oldugu surece kendi transaction rate'i etrafinda rasgele satin alma yapmaya devam eder.
#Transcation rateler her bir musteriye gore degisir ve tum kitle icin gamma dagilir.(r,a) (ONEMLI)

#Dropout Process(Till you die)
#Her bir musterinin p olasiligi ile dropout rate(dropout probability)'i vardir.
#Bir musteri alisveris yaptiktan sonra belirli bir olasilikla drop olur.
#Drop rateler her bir musteriye gore degisir ve tum kitle icin beta dagilir.(a,b) (ONEMLI)

#CLTV = BG/NBD Model * Gamma Gamma Submodel

#Gamma Gamma Submodel
#Bir musterinin islem basina ortalama ne kadar kar getirecegini tahmin etmek icin kullanilir.
#Bir musterinin islemlerinin parasal degeri(monetary) transaction valuelarin ortalamasi etrafinda rastgele dagilir.
#Ortalama transaction value, zaman icinde kullanicilar arasinda degisebilir fakat tek bir kullanici icin degismez.
#Ortalama transaction value tum musteriler arasinda gamma dagilir.

#CLTV Prediction with BG-NBD & Gamma Gamma (BG-NBD ve Gamma Gamma ile CLTV Tahmini)

#1-)Data Preperation(Verinin Hazirlanmasi)
#2-)BG-NBD ile Expected Number of Trasnsaction
#3-)Gamma-Gamma Modeli ile Expected Average Profit
#4-)BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanmasi
#5-)CLTV'ye Gore Segmentlerin Olusturulmasi
#6-)Calismanin Fonksiyonlastirilmasi

#Bir e-ticaret sirketi musterilerini segmentlere gore pazarlama stratejileri belirlemek istiyor.

#Veri Setinin Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli online bir satis magazasinin 01/12/2009 - 09/12/2011 tarihleri
#arasindaki satislarini iceriyor.

#Degiskenler
# InvoiceNo: Fatura numarasi. Her isleme yani faturaya ait essiz numara. C ile basliyorsa iptal edilen islem.
# StockCode: Urun kodu. Her bir urun icin essiz numara.
# Description: Urun ismi
# Quantity: Urun adedi. Faturalardaki urunlerden kacar tane satildigini ifade eder.
# InvoiceDate: Fatura tarihi ve zamani.
# UnitPrice: Urun fiyati (Sterlin)
# CustomerID: Essiz musteri numarasi.
# Country: Musterinin yasadigi ulke.

#Gerekli Kutuphane ve Fonksiyonlar

#pip install lifetimes, !pip install lifetimes(python konsoldan bu sekilde)
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format',lambda x: '%4f' % x)
from sklearn.preprocessing import MinMaxScaler #(0-1 ya da 0-100 gibi degerler arasina cekmek istersek bunu kullaniriz.)


#Once aykiri degerleri tespit edecegiz ardindan bu degerleri baskilayacagiz(silmeyecegiz)
def outlier_thresholds(dataframe, variable):
    #gorevi: kendisine girilen degisken icin esik deger belirlemektir.
    quartile1 = dataframe[variable].quantile(0.01)
    #burada olmasi gereken degerler 0.25 ve 0.75 iken bu kisisel yorumumuz oldugundan bu degerleri yazdik.
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Verinin Okunmasi

df_ = pd.read_excel("Functions_conditions_loop_comprehensions/Projects_doc/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

#Veri On Isleme

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

#Prepation of Lifetime Data Structure(Lifetime Veri Yapisinin Hazirlanmasi)

#recency: Son satin alma uzerinden gecen zaman. Haftalik. (kullanici ozelinde)
#T: Musterinin yasi. Haftalik. (analiz tarihinden ne kadar sure once ilk satin alma yapilmis)
#frequency: tekrar eden toplam satin alma sayisi (frequencty > 1)
#monetary_value: satin alma basina ortalama kazanc (burada gormeye alisik oldugumuz degerler oncekileriyle ayni anlamda degil.)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         "Invoice": lambda num: num.nunique(),
                                         "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0) #sutunlarin en ustundeki yazilari siler gibi dusun.

cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df.describe().T

cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

##2-) BG-NBD Modelinin Kurulmasi

bgf = BetaGeoFitter(penalizer_coef=0.001)                 #Bir model nesnesi araciligla fit metodunu kullanarak frequency recency ve
                                                           #musteri yasi degerini verdigimizde modeli kurmus olacak.
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

## 1 hafta icerisinde en cok satin alma bekledigimiz 10 musteri kimdir? ##
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                           cltv_df["frequency"],
                                                           cltv_df["recency"],
                                                           cltv_df["T"]).sort_values(ascending=False).head(10)

bgf.predict(1,                                                        #BG-NBD modeli icin bu metod gecerlidir. Ancak
               cltv_df["frequency"],                                     #Gamma Gamma modeli icin gecerli degildir.
               cltv_df["recency"],
               cltv_df["T"]).sort_values(ascending=False).head(10)
cltv_df["expected_purc_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                                             cltv_df["frequency"],
                                                                                             cltv_df["recency"],
                                                                                             cltv_df["T"])

## 1 ay icerisinde en cok satin alma bekledigimiz 10 musteri kimdir? ##
bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                           cltv_df["frequency"],
                                                           cltv_df["recency"],
                                                           cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                                                             cltv_df["frequency"],
                                                                                             cltv_df["recency"],
                                                                                             cltv_df["T"])


##Bir ay icerisinde ne kadar satin alma olur?

bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                           cltv_df["frequency"],
                                                           cltv_df["recency"],
                                                           cltv_df["T"]).sum()

## 3 ay icerisinde en cok satin alma bekledigimiz 10 musteri kimdir? ##

bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sum()
cltv_df["expected_purc_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                                           cltv_df["frequency"],
                                                                                           cltv_df["recency"],
                                                                                           cltv_df["T"])


###  Tahmin Sonuclarinin Degerlendirilmesi  ###
plot_period_transactions(bgf)
plt.show()

##3-)GAMMA-GAMMA Modelinin Kurulmasi

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).head(10)

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"]).sort_values(ascending=False)

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##Calculating of CLTV with BG-NBD and GG Model(BG-NBD ve GG Modeli ile CLTV'nin Hesaplanmasi)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time = 3,               #3 aylik
                                   freq = "W",             #T'nin frekans bilgisi
                                   discount_rate = 0.01)

cltv.head()

cltv = cltv.reset_index()

cltv.drop("index", axis=1, inplace=True)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

#Senin duzenli ortalama islem kapasitesi olan musterin eger churn olmadiysa musterinin recency'si arttikca satin alma
#olasiligi yukselir.

##5-)CLTV'ye Gore Segmentlerin Olusturulmasi

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.groupby("segment").agg({"count", "mean", "sum"})

##6-) Functionalization

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

df = df_.copy()
cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")







































