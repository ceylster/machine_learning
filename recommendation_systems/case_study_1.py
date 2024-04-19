# Association Rule Based ecommender System
# Turkiye’nin en buyuk online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri
# bulusturmaktadir. Bilgisayarin veya akilli telefonunun uzerinden birkac dokunusla temizlik,tadilat,
# nakliyat gibi hizmetlere kolayca ulasilmasini saglamaktadir.
# Hizmet alan kullanicilari ve bu kullanicilarin almis olduklari servis ve kategorileri iceren veri setini
# kullanarak Association Rule Learning ile urun tavsiye sistemi olusturulmak istenmektedir.

import pandas as pd
import datetime
from mlxtend.frequent_patterns import apriori, association_rules

# Gorev 1: Veriyi Hazirlama

# Adim 1: armut_data.csv dosyasini okutunuz.

df = pd.read_csv('Miull/Projects_doc/armut_data.csv')
df.isnull().sum()
# Adim 2: ServisID her bir CategoryID ozelinde farkli bir hizmeti temsil etmektedir. ServiceID ve CategoryID’yi "_"
# ile birlestirerek bu hizmetleri temsil edecek yeni bir değisken olusturunuz.

df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# Adim 3: Veri seti hizmetlerin alindigi tarih ve saatten olusmaktadir, herhangi bir sepet tanimi(fatura vb.)
# bulunmamaktadir. Association Rule Learning uygulayabilmek icin bir sepet (fatura vb.) tanimi olusturulmasi
# gerekmektedir. Burada sepet tanimi her bir musterinin aylik aldigi hizmetlerdir. Ornegin; 7256 id'li müşteri
# 2017'in 8. ayinda aldigi 9_4,  46_4 hizmetleri bir sepeti; 2017’in 10.ayinda aldigi 9_4, 38_4  hizmetleri
# baska bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanimlanmasi gerekmektedir. Bunun icin oncelikle
# sadece yil ve ay iceren yeni bir date degiskeni olusturunuz. UserID ve yeni olusturdugunuz date degiskenini "_" ile
# birlestirirek ID adinda yeni bir değiskene atayiniz.

df["New_Date"] = pd.to_datetime(df["CreateDate"])
df["New_Date"] = df["New_Date"].dt.strftime("%Y-%m")
df["SepetID"] = df["UserId"].astype(str) + "_" + df["New_Date"]

# Gorev 2: Birliktelik Kurallari Uretiniz ve Oneride bulununuz

# Adim 1: sepet, hizmet pivot table’i olusturunuz.

df_arl = df.pivot_table(index="SepetID", columns="Hizmet", values={"CategoryId": "count"})
df_arl.head()
df_arl = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Adim 2: Birliktelik kurallarini olusturunuz.

frequent_itemsets = apriori(df_arl, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]
arl_recommender(rules,"2_0", 4)