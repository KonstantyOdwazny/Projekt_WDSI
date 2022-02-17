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
        path_f = Path(str(img_path) + root.find("./filename").text)
        data_path.append(path_f)
        # Odczyt nazwy zdjecia
        filename = root.find("./filename").text
        data_filename.append(str(filename))
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
    """
    Wyswietla pojedynczy obraz wraz z ramka oraz predykowana etykieta
    """
    image = read_image(paths)
    show_labeled_image(image, boxes, labels)
def get_file_list(root, file_type):
    """
    Funkcja pobiera typ pliku w tym przypadku xml oraz odsylacz do niego root i zwraca xml w formie listy
    """
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
def draw_grid(images, row, col, h, w):
    """
    Funkcja ktora tworzy ramki dla wyswietlania obrazow
    Zwraca tablice wszystkich 24 obrazow dla przyjetej listy obrazow image
    """
    h_size = int(h / row)
    w_size = int(w / col)
    image_all = np.zeros((h, w, 4), dtype=np.uint8)
    r = 0
    c = 0
    if len(images) <= 24:
        n = len(images)
    else:
        n = 24
    for cur_image in range(n):
        if c < col and r < row:
            img = cv2.imread(images[cur_image], cv2.IMREAD_UNCHANGED)
            image_resized = cv2.resize(img, (w_size, h_size))
            image_all[r * h_size: (r + 1) * h_size, c * w_size: (c + 1) * w_size, :] = image_resized
            r += 1
        else:
            if r >= row:
                r = 0
                c += 1

    return image_all
def display_images(corr, incorr, za_male):
    """
    Funkcja ktora wysiwetla ramki z obrazami dla obrazow:
    corr - poprawnie wykrytych
    incorr - niepoprawnie wykrytych
    za_male - poprawnie wykrytych ale odrzucone przez zbyt maly rozmiar znaku
    """

    image_corr = draw_grid(corr, 8, 3, 800, 600)
    image_incorr = draw_grid(incorr, 8, 3, 800, 600)
    image_za_male = draw_grid(za_male, 8, 3, 800, 600)

    cv2.imshow('images correct', image_corr)
    cv2.imshow('images incorrect', image_incorr)
    cv2.imshow('Za male', image_za_male)
    cv2.waitKey()

    return

def test_and_evaluate(ilosc, path_f, filename, real_labels, model):
    """
    :param ilosc: stała wartość int oznacza ilość obrazów
    :param path_f: wektor ścieżek do obrazów
    :param filename: wektor z nazwami obrazów
    :param real_labels: etykiety prawdziwe dla zbioru testowego ( znane ale nie biora udziału w predykcji),
    służą do ewaluacji
    :param model: Wyuczony model
    :return: Precision - precyzje ewaluacji (predykcji obiektow)
    Funkcja służy do testowania i ewaluacji błędu predykcji dla danego modelu
    """
    # ann_path_list = get_file_list(ann_path, '.xml')  #otwarcie listy plikow xml
    TP = 0      #Liczba prawdziwie pozytywnych zdjec
    FN = 0      #Liczba falszywie pozytywnych zdjec
    corr = []   #Lista wykrytych zdjec
    in_corr = []    #Lista niepoprawnie wykrytych obrazow
    wrong_size = [] #Lista obrazow poprawnie wykrytych ale o zlym rozmiarze

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

        image_stats(filename[id], i, xmin_list, xmax_list, ymin_list, ymax_list)

        pre_labels = []
        for s in range(len(scores)):
            dlugosc_znaku = xmax_list[s] - xmin_list[s]
            wysokosc_znaku = ymax_list[s] - ymin_list[s]
            if dlugosc_znaku / width >= 0.1 and wysokosc_znaku / height >= 0.1:
                pre_labels.append(predict_labels[s])

        if 'crosswalk' in pre_labels and 'crosswalk' in real_labels[id]:
            TP += 1
            # display(str(path_f[id]), boxes[s], predict_labels[s]) # wyswietlanie poprawnych zdjec
            corr.append(str(path_f[id]))
        elif 'crosswalk' in predict_labels and 'crosswalk' in real_labels[id] and 'crosswalk' not in pre_labels:
            wrong_size.append(str(path_f[id]))
        elif 'crosswalk' not in predict_labels and 'crosswalk' in real_labels[id] and 'crosswalk' not in pre_labels:
            FN += 1
            in_corr.append(str(path_f[id]))
        elif 'crosswalk' in pre_labels and 'crosswalk' not in real_labels[id]:
            FN += 1
            in_corr.append(str(path_f[id]))

    if TP + FN != 0:
        Precision = TP / (TP + FN)

    print("Lista wykrytych zdjec:", corr)
    print("Wykryte zdjecia:")
    display_images(corr, in_corr, wrong_size)
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
    print("Testowanie: ")
    score = test_and_evaluate(liczba_obrazow, test_path, test_filename, test_labels, model)
    print("Precision = ", score)
    return
if __name__ == '__main__':
    main()











