import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os
import cv2
import csv
import bs4
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from detecto.utils import read_image
from detecto.core import Dataset
from detecto.visualize import show_labeled_image
from detecto.core import DataLoader, Model
from detecto import core, utils


def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]
def test(ann_path, img_path, model):
    """
    Otwarcie pliku xml i wydobycie z nich potrzebnych informacji jak
    Filename, patch, xmin,xmax,ymin,ymax,width, height, class
    """
    ann_path_list = get_file_list(ann_path, '.xml')
    data = []
    TP = 0
    FN = 0
    for a_path in ann_path_list:
        root = ET.parse(a_path).getroot()

        path_f = Path(str(img_path) + '/' + root.find("./filename").text)
        width = int(root.find("./size/width").text)
        height = int(root.find("./size/height").text)
        filename = root.find("./filename").text
        i = 0
        predict_labels = []
        for cl in root.findall("./object/name"):
            # print(cl.text)
            predict_labels.append(cl.text)
            i = i + 1
        xmin_list = []
        for xmin in root.findall("./object/bndbox/xmin"):
            # print(xmin.text)
            xmin_list.append(int(xmin.text))
        ymin_list = []
        for ymin in root.findall("./object/bndbox/ymin"):
            # print(ymin.text)
            ymin_list.append(int(ymin.text))
        xmax_list = []
        for xmax in root.findall("./object/bndbox/xmax"):
            # print(xmax.text)
            xmax_list.append(int(xmax.text))
        ymax_list = []
        for ymax in root.findall("./object/bndbox/ymax"):
            # print(ymax.text)
            ymax_list.append(int(ymax.text))

        print("Filename:" , filename)
        print("n:", i)
        for x in range(len(ymax_list)):
            print("xmin:",xmin_list[x],"xmax:",xmax_list[x],"ymin:",ymin_list[x],"ymax:",ymax_list[x])
        """
        Predykcja
        """
        image = read_image(str(path_f))
        labels, ramki, wyniki = model.predict(image)

        """ Ewaluacja """
        score = max(wyniki)
        #Najwieksza bedzie zawsze wartosc dla wyniki[0]
        pom_tp = TP
        for j in range(len(predict_labels)):
            dlugosc_znaku = xmax_list[j] - xmin_list[j]
            wysokosc_znaku = ymax_list[j] - ymin_list[j]
            #warunek ktory sprawdza czy dlugosc i wyskosc znaku zajmuja przynajmniej 1/10 zdjecia
            if dlugosc_znaku/width >= 0.1 and wysokosc_znaku/height >= 0.1:
                if predict_labels[j] == labels[0]:
                    if score >= 0.6:
                        TP = TP + 1
                    break
        if pom_tp == TP:
            FN = FN + 1












def main():
    # class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
    base_path = 'train/'
    ann_path = base_path + 'annotations/'
    img_path = base_path + 'images/'
    test_base_path = 'test/'
    test_ann_path = test_base_path + 'annotations/'
    test_img_path = test_base_path + 'images/'
    """
    Uzywane tylko podczas tworzenia plikow csv :
    """
    # data_train = get_train_df(ann_path, img_path)
    # data_test = get_train_df(test_ann_path,test_img_path)

    dataset = Dataset(ann_path, img_path)

    # Testowanie działania biblioteki
    # image, targets = dataset[10]
    # show_labeled_image(image, targets['boxes'], targets['labels'])

    labels = ['speedlimit', 'stop', 'crosswalk', 'trafficlight']
    """
    Uzywane tylko raz podczas trenowania
    """
    # model = Model(labels)
    # model.fit(dataset)
    # model.save('sings_rmodel.pth')

    model = core.Model.load('sings_rmodel.pth', labels)
    test(test_ann_path, test_img_path, model)
    return
if __name__ == '__main__':
    main()











