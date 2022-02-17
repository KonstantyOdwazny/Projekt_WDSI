import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
import cv2
from detecto.utils import read_image
from detecto.core import Dataset
from detecto.core import DataLoader, Model
from detecto import core, utils
from detecto.visualize import show_labeled_image
import PIL

def load(ann_path,img_path):
    """
    Funkcja służąca do odczytu danych z pliku xml
    """
    # otwarcie listy plikow xml
    ann_path_list = get_file_list(ann_path, '.xml')
    ilosc_obrazow = 0
    data_path = []
    data_filename = []
    data_labels = []
    for a_path in ann_path_list:
        ilosc_obrazow = ilosc_obrazow + 1
        root = ET.parse(a_path).getroot()
        # Odczytywanie sciezki do zdjecia
        path_f = Path(str(img_path) + '/' + root.find("./filename").text)
        data_path.append(path_f)
        # Odczyt nazwy zdjecia
        filename = root.find("./filename").text
        data_filename.append(filename)
        #Odnajdywanie wszystkich etykiet klas dla danego obrazka
        labels = []
        for cl in root.findall("./object/name"):
            labels.append(cl.text)
        data_labels.append(labels)

    return ilosc_obrazow, data_path, data_filename, data_labels
def train_model(labels,dataset):
    """
    Funkcja ktora przyjmuje slownik klas labels oraz dane sciezki do etykiet i obrazow
    Tworzy model za pomoca biblioteki detecto
    """
    model = Model(labels)
    model.fit(dataset)
    model.save('sings_rmodel.pth')
    return

def display(paths, boxes, labels):
    image = read_image(paths)
    show_labeled_image(image, boxes, labels)
def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]
def image_stats(filename, i, xmin, xmax, ymin, ymax):
    """
    Funkcja do wyswietlania statysyk znalezionych obrazow wedlug wzorca:
    file_1
    n_1
    xmin_1_1 xmax_1_1 ymin_1_1 ymax_1_1
    Gdzie :
    file_1 - nazwa 1. zdjęcia,
    n_1 - liczba obiektów wykrytych na 1. zdjęciu,
    xmin_1_1 xmax_1_1 ymin_1_1 ymax_1_1 - współrzędne prostokąta zawierającego pierwszy wykryty obiekt na 1. zdjęciu.
    """
    print("Filename:", filename)
    print("n:", i)
    for x in range(len(ymax)):
        print("xmin:", xmin[x], "xmax:", xmax[x], "ymin:", ymin[x], "ymax:", ymax[x])

    return
def test(ilosc, path_f, filename, real_labels, model):
    """
    Testowanie modelu
    """
    # ann_path_list = get_file_list(ann_path, '.xml')  #otwarcie listy plikow xml
    TP = 0 #Liczba prawdziwie pozytywnych zdjec
    FN = 0 #Liczba falszywie pozytywnych zdjec
    corr = [] #Lista wykrytych zdjec

    Precision = 0
    for id in range(ilosc):
        # Zmienne pomocniczne
        xmin_list = []
        ymin_list = []
        xmax_list = []
        ymax_list = []
        i = 0
        # #Wydobywanie wielkosci danego obrazu nie korzystajac z plikow xml
        size_image = PIL.Image.open(str(path_f[id]))
        width, height = size_image.size
        """
        Predykcja
        """
        predict_labels = []
        boxes = []
        scores = []
        image = utils.read_image(str(path_f[id])) #odczyt obrazka z sciezki
        klasy, ramki, wyniki = model.predict(image) #funkcja sluzaca do obliczenia wartosci prawdopodobienstwa wystepowania znaku o danej wielkosci i klasie
        # print("predict_labels", klasy)
        # print("boxes", ramki)
        # print("scores", wyniki)
        for j in range(len(klasy)):
            if klasy is not None and ramki is not None and wyniki is not None:
                if wyniki[j] >= 0.6:
                    predict_labels.append(klasy[j])
                    scores.append(wyniki[j])
                    boxes.append(ramki[j])
                    i = i + 1
                    xmin_list.append(int(ramki[j][0]))
                    xmax_list.append(int(ramki[j][2]))
                    ymin_list.append(int(ramki[j][1]))
                    ymax_list.append(int(ramki[j][3]))

        # print("Filename:", filename[id])
        # print("n:", i)
        # for x in range(len(ymax_list)):
        #     print("xmin:", xmin_list[x], "xmax:", xmax_list[x], "ymin:", ymin_list[x], "ymax:", ymax_list[x])
        image_stats(filename[id], i, xmin_list, xmax_list, ymin_list, ymax_list)

        """ Ewaluacja oraz wyswietlanie poprawnie sklasyfikowanych obrazow """
        score = [] # Wynik prawdopodobienstwa wystepowania znaku z danej klasy
        indeks = []
        for s in range(len(scores)):
            if predict_labels[s] == 'crosswalk':
                dlugosc_znaku = xmax_list[s] - xmin_list[s]
                wysokosc_znaku = ymax_list[s] - ymin_list[s]
                if dlugosc_znaku/width >= 0.1 and wysokosc_znaku/height >= 0.1:
                    score.append(scores[s])
                    indeks.append(s)


        pom_tp = TP
        if indeks is not None:
            for j in range(len(indeks)):
                for z in range(len(real_labels[id])):
                    if real_labels[id][z] == predict_labels[indeks[j]]:
                        TP = TP + 1
                        display(str(path_f[id]),boxes[indeks[j]],predict_labels[indeks[j]]) # wyswietlanie poprawnych zdjec
                        corr.append(filename[id])
                if pom_tp == TP:
                    FN = FN + 1

    Precision = TP/(TP + FN)
    print("Lista wykrytych zdjec:", corr)
    return Precision

def main():
    """
    Sciezki dostepu do plików
    """
    base_path = 'train/'
    ann_path = base_path + 'annotations/'
    img_path = base_path + 'images/'
    test_base_path = 'test/'
    test_ann_path = test_base_path + 'annotations/'
    test_img_path = test_base_path + 'images/'

    # Słownik klas
    labels = ['speedlimit', 'stop', 'crosswalk', 'trafficlight']

    # Tworzenie sciezki do danych etykiet oraz obrazow dla wyuczenia modelu
    dataset = Dataset(ann_path, img_path)

    """
    Uzywane tylko raz podczas trenowania
    """
    # print("Trenowanie")
    # train_model(labels,dataset)

    # Wczytanie wyuczonego modelu
    model = core.Model.load('sings_rmodel.pth', labels)

    liczba_obrazow, test_path, test_filename, test_labels = load(test_ann_path, test_img_path)
    score = test(liczba_obrazow, test_path, test_filename, test_labels, model)
    print("Score = ", score)
    return
if __name__ == '__main__':
    main()











