# Advanced Tree-Based
# Birden cok karar agacinin urettigi tahminlerin bir araya getirilerek degerlendirilmesine dayanir.
# Bagging(Breiman, 1996) ile Random Subsplace (Ho, 1998) yontemlerinin birlesimi ile olusmwustur.
# Agaclarr icin gozlemler bootstrap rastgele ornek secim yontemi ile degiskenler random subspace yontemi ile secilir.
# Karar agacinin her bir dugumunde en iyi dallara ayirici (bilgi kazanci) degisken tum degiskenler arasindan
# rastgele secile daha az sayidaki degisken arasindan secilir.
# Agac olusturmada veri setinin 2/3'u kullanilir. Disarida kalan veri agaclarin performans degerlendirmesi ve degisken
# oneminin belirlenmesi icin kullanilir.
# Her dugum noktasinda rastgele degisken secimi yapilir. (regresyonda p/3, siniflamada (p)^1/2). Fakat biz bunu hiper
# parametre optimizasyonunda her dugum noktasinda kac tane degiskeni goz onunde bulundurmasi gerektigi gorevini
# hiperparametre optimizaasyonu ile belirlemis olacagiz.
# Bagging yonteminin kilit noktasi boostrap rastgele orneklem yontemidir. Bagging yonteminde agaclarin birbirlerine
# bagliliklari yoktur. Boosting yonteminde ise agaclar artiklar uzerine kurulur. Dolayisiyla agaclarin birbirlerine
# bagimliliklari vardir.
