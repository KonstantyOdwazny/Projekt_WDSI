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


def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]
def get_train_df(ann_path, img_path):
    ann_path_list = get_file_list(ann_path, '.xml')
    ann_list = []
    data = []
    for a_path in ann_path_list:
        root = ET.parse(a_path).getroot()

        path_f = Path(str(img_path) + '/' + root.find("./filename").text)
        width = root.find("./size/width").text
        height = root.find("./size/height").text
        filename = root.find("./filename").text

        i = 0
        cl_list = []
        for cl in root.findall("./object/name"):
            # print(cl.text)
            cl_list.append(cl.text)
            i = i + 1
        xmin_list = []
        for xmin in root.findall("./object/bndbox/xmin"):
            # print(xmin.text)
            xmin_list.append(xmin.text)
        ymin_list = []
        for ymin in root.findall("./object/bndbox/ymin"):
            # print(ymin.text)
            ymin_list.append(ymin.text)
        xmax_list = []
        for xmax in root.findall("./object/bndbox/xmax"):
            # print(xmax.text)
            xmax_list.append(xmax.text)
        ymax_list = []
        for ymax in root.findall("./object/bndbox/ymax"):
            # print(ymax.text)
            ymax_list.append(ymax.text)

        class_dict = {'speedlimit': 0, 'stop': 0, 'crosswalk': 1, 'trafficlight': 0}
        #print(cl_list, i)
        class_id = 0
        for c in cl_list:
            id = class_dict[c]
            if id != 0:
                class_id = id
                break
        image = cv2.imread(os.path.join('./', path_f))
        data.append({'image': image, 'label': class_id})

    return data


base_path = 'train/'
ann_path = base_path + 'annotations/'
img_path = base_path + 'images/'
test_base_path = 'test/'
test_ann_path = test_base_path + 'annotations/'
test_img_path = test_base_path + 'images/'
"""
Uzywane tylko podczas tworzenia plikow csv :
"""
data_train = get_train_df(ann_path, img_path)
data_test = get_train_df(test_ann_path,test_img_path)



def learn_bovw(data):

    dict_size = 128
    bow = cv2.BOWKMeansTrainer(dict_size)

    sift = cv2.SIFT_create()
    for sample in data:
        kpts = sift.detect(sample['image'], None)
        kpts, desc = sift.compute(sample['image'], kpts)

        if desc is not None:
            bow.add(desc)

    vocabulary = bow.cluster()

    np.save('voc.npy', vocabulary)
def extract_features(data):

    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    vocabulary = np.load('voc.npy')
    bow.setVocabulary(vocabulary)
    for sample in data:
        # compute descriptor and add it as "desc" entry in sample
        # TODO PUT YOUR CODE HERE
        kpts = sift.detect(sample['image'], None)
        desc = bow.compute(sample['image'], kpts)
        sample['desc'] = desc
        # ------------------

    return data

# class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
# print('learning BoVW')
# learn_bovw(data_train)


