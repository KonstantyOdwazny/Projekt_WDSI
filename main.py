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


def display_dataset_stats(data):
    class_to_num = {}
    for idx, sample in enumerate(data):
        class_id = sample['label']
        if class_id not in class_to_num:
            class_to_num[class_id] = 0
        class_to_num[class_id] += 1

    class_to_num = dict(sorted(class_to_num.items(), key=lambda item: item[0]))
    # print('number of samples for each class:')
    print(class_to_num)
def balance_dataset(data, ratio):
    sampled_data = random.sample(data, int(ratio * len(data)))

    return sampled_data

# class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
#data_train = load_data('./', 'Train.csv')
print('train dataset before balancing:')
display_dataset_stats(data_train)
data_train = balance_dataset(data_train, 1.0)
print('train dataset after balancing:')
display_dataset_stats(data_train)

#data_test = load_data('./', 'Test.csv')
print('test dataset before balancing:')
display_dataset_stats(data_test)
data_test = balance_dataset(data_test, 1.0)
print('test dataset after balancing:')
display_dataset_stats(data_test)


