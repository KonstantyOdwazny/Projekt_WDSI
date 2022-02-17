# Projekt_WDSI
 Projekt zaliczeniowy z przedmiotu Wprowadzenie do Sztucznej Inteligencji
 Założeniem Projektu było rozpoznawanie znaków przejścia dla pieszych, które zajmują co najmniej 1/10 szerokości i wysokości zdjęcia 
 Kroki podejmowane podczas wykonania zadania:
 1. Podzieliłem zbiór danych na testowy oraz treningowy i stworzyłem foldery train i test oraz pod foldery dla obu: annotations i images.
 2. Wytrenowałem sieć za pomocą biblioteki detecto(https://detecto.readthedocs.io/en/latest/). Sieć trenowałem przy użyciu Google Colab gdyż tam można użyć zasobów GPU,
 podczas gdy w Pycharmie nie, co znacznie spowalniało uczenie się modelu. (W Google Colab zajeło około 20 min przy zbiorze uczącym mającym 518 zdjęć) Link do Google Colab:
(https://colab.research.google.com/drive/1e2xQiSiGT35oxAV-SlGzwPOu548ywYRO?usp=sharing)
Link do pobrania wytrenowanego przeze mnie modelu: (https://drive.google.com/file/d/1HDrZstTJ-L8pB_8w5Hl--mpoFvmjKPzd/view?usp=sharing)
3. Następnie wczytałem plik z modelem do Pycharma i przeszedłem do testowania. Podczas testowania pobieram z zdjęcia jego rozmiary: width i height. Następnie za pomocą 
funkcji predict z biblioteki detecto otrzymuje wektory: klas(labels), ramek(boxes) oraz wyniki prawdopodobieństwa wystąpienia danej klasy(scores). W wektorze predykowanym 
ramki(boxes) mamy informację o xmin,ymin,xmax,ymax. Zatem przyjąłem założenie, że dany znak można przyjąć, że występuje na zdjęciu, kiedy prawdopodobieństwo jego wystąpienia jest 
większe niż 80%(0.8), więc dla tych klas(labels), dla których to założenie jest prawdą wypisuje jej położenie i zliczam ilość występujących obiektów na zdjęciu. Jak można zauważyć
,mamy tutaj możliwość manipulacji dokładnością wyszukiwania w postaci zmiany dopuszczalnego progu prawdopodobieństwa występowania danego znaku na obrazie.
4. Następnym krokiem było wypisanie informacji o obrazie według zasad nakreślonych w opisie projektu.
5. Po wypisaniu informacji o danym obrazie, sprawdzałem czy znajduje się na nim poszukiwany znak przejścia dla pieszych oraz czy zajmuje ono przynajmniej 1/10 szerokości i  
wysokości zdjęcia. Sprawdzane są następujące warunki:
- Jeśli znak przejścia dla pieszych występuje w wektorze predykowanych klas oraz występuje rzeczywiście na obrazie co wiemy odgórnie( podczas predykcji nie używamy plików xml
tylko tutaj już dokonujemy ewaluacji poprawności naszego wyszukiwania) to : Wyświetl obraz wraz z jego ramkami i opisami, przypisz go do wektora z poprawnie znalezionymi 
zdjęciami oraz zwiększ współczynnik TP (prawdziwie poprawnie znalezionych zdjęć).
- Jeśli znak przejścia dla pieszych znajduję się w wektorze predykowanym oraz znajduje się naprawdę na obrazie, ale jest niewłaściwych rozmiarów to: przypisz go do wektora z 
zdjęciami poprawnymi ale o za małych niż założony rozmiarze.
- Jeśli znak przejścia dla pieszych nie znajduje się w wektorze predykowanym ale występuje tak naprawdę na zdjęciu to: zwiększ współczynnik FN(fałszywie zidentyfikowany) oraz 
dodaj to zdjęcie do wektora z zdjęciami niepoprawnie znalezionymi.
- Jeśli znak przejścia dla pieszych znajduje się w wektorze predykowanym ale nie występuje tak naprawdę na zdjęciu to: zwiększ współczynnik FN(fałszywie zidentyfikowany) oraz 
dodaj to zdjęcie do wektora z zdjęciami niepoprawnie znalezionymi.
6. Następnie po zakończeniu pętli for z wszystkimi testowanymi zdjęciami obliczam precyzję znajdywania znaków przejść dla pieszych według wzoru: Precision = TP/(Tp + FN)
7. Ostatnim etapem jest wyświetlenie obrazów z podziałem na 3 zbiory: Poprawnie znalezione(image correct), Niepoprawnie znalezione(image incorrect) oraz Za małe, żeby być 
poprawnymi

