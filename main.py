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


def display(paths, boxes, labels):
    image = read_image(paths)
    show_labeled_image(image, boxes, labels)
def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]
def test(ann_path, img_path, model):
    """
    Otwarcie pliku xml i wydobycie z nich potrzebnych informacji jak rzeczywiste klasy znakow
    Zdobycie informacji : Filename, patch, xmin,xmax,ymin,ymax,width, height, class
    """
    ann_path_list = get_file_list(ann_path, '.xml')  #otwarcie listy plikow xml
    TP = 0 #Liczba prawdziwie pozytywnych zdjec
    FN = 0 #Liczba falszywie pozytywnych zdjec
    corr = [] #Lista wykrytych zdjec

    Precision = 0
    for a_path in ann_path_list:
        root = ET.parse(a_path).getroot()  #odczyt pliku xml

        path_f = Path(str(img_path) + '/' + root.find("./filename").text)
        size_image = PIL.Image.open(str(path_f))
        width, height = size_image.size
        # width = int(root.find("./size/width").text)
        # height = int(root.find("./size/height").text)
        filename = root.find("./filename").text
        i = 0
        predict_labels = [] #lista predykowanych klas
        for cl in root.findall("./object/name"):
            # print(cl.text)
            predict_labels.append(cl.text)
        xmin_list = []
        # for xmin in root.findall("./object/bndbox/xmin"):
        #     # print(xmin.text)
        #     xmin_list.append(int(xmin.text))
        ymin_list = []
        # for ymin in root.findall("./object/bndbox/ymin"):
        #     # print(ymin.text)
        #     ymin_list.append(int(ymin.text))
        xmax_list = []
        # for xmax in root.findall("./object/bndbox/xmax"):
        #     # print(xmax.text)
        #     xmax_list.append(int(xmax.text))
        ymax_list = []
        # for ymax in root.findall("./object/bndbox/ymax"):
        #     # print(ymax.text)
        #     ymax_list.append(int(ymax.text))
        """
        Predykcja
        """
        labels = []
        boxes = []
        scores = []
        image = read_image(str(path_f)) #odczyt obrazka z sciezki
        klasy, ramki, wyniki = model.predict(image) #funkcja sluzaca do obliczenia wartosci prawdopodobienstwa wystepowania znaku o danej wielkosci i klasie
        # print("labels", klasy)
        # print("boxes", ramki)
        # print("scores", wyniki)
        for j in range(len(klasy)):
            if wyniki[j] >= 0.6:
                labels.append(klasy[j])
                scores.append(wyniki[j])
                boxes.append(ramki[j])
                i = i + 1
                xmin_list.append(int(ramki[j][0]))
                xmax_list.append(int(ramki[j][2]))
                ymin_list.append(int(ramki[j][1]))
                ymax_list.append(int(ramki[j][3]))

        print("Filename:", filename)
        print("n:", i)
        for x in range(len(ymax_list)):
            print("xmin:", xmin_list[x], "xmax:", xmax_list[x], "ymin:", ymin_list[x], "ymax:", ymax_list[x])

        """ Ewaluacja oraz wyswietlanie poprawnie sklasyfikowanych obrazow """
        score = [] # Wynik prawdopodobienstwa wystepowania znaku z danej klasy
        indeks = []
        for s in range(len(scores)):
            if labels[s] == 'crosswalk':
                dlugosc_znaku = xmax_list[s] - xmin_list[s]
                wysokosc_znaku = ymax_list[s] - ymin_list[s]
                if dlugosc_znaku/width >= 0.1 and wysokosc_znaku/height >= 0.1:
                    score.append(scores[s])
                    indeks.append(s)


        pom_tp = TP
        if indeks is not None:
            for j in range(len(indeks)):
                for z in range(len(predict_labels)):
                    if predict_labels[z] == labels[indeks[j]]:
                        TP = TP + 1
                        display(str(path_f),boxes[indeks[j]],labels[indeks[j]]) # wyswietlanie poprawnych zdjec
                        corr.append(filename)
                if pom_tp == TP:
                    FN = FN + 1



    if TP != 0 and FN != 0:
        Precision = TP / (TP + FN)
    else:
        Precision = 0

    print("Lista wykrytych zdjec:", corr)
    return Precision


def main():
    base_path = 'train/'
    ann_path = base_path + 'annotations/'
    img_path = base_path + 'images/'
    test_base_path = 'test/'
    test_ann_path = test_base_path + 'annotations/'
    test_img_path = test_base_path + 'images/'

    labels = ['speedlimit', 'stop', 'crosswalk', 'trafficlight']
    """
    Uzywane tylko raz podczas trenowania
    """
    # dataset = Dataset(ann_path, img_path)
    # model = Model(labels)
    # model.fit(dataset)
    # model.save('sings_rmodel.pth')

    model = core.Model.load('sings_rmodel.pth', labels)
    score = test(test_ann_path, test_img_path, model)
    print("Score = ", score)
    return
if __name__ == '__main__':
    main()











