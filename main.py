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


def get_file_list(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]
def get_train_df(ann_path, img_path):
    ann_path_list = get_file_list(ann_path, '.xml')
    ann_list = []
    with open('Train.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Width', 'Height', 'Roi.X1' , 'Roi.Y1' , 'Roi.X2' , 'Roi.Y2' , 'ClassId' , 'Path'])
        for a_path in ann_path_list:
            root = ET.parse(a_path).getroot()
            ann = {}
            ann['filename'] = Path(str(img_path) + '/' + root.find("./filename").text)
            ann['width'] = root.find("./size/width").text
            ann['height'] = root.find("./size/height").text
            ann['class'] = root.find("./object/name").text
            ann['xmin'] = int(root.find("./object/bndbox/xmin").text)
            ann['ymin'] = int(root.find("./object/bndbox/ymin").text)
            ann['xmax'] = int(root.find("./object/bndbox/xmax").text)
            ann['ymax'] = int(root.find("./object/bndbox/ymax").text)
            ann_list.append(ann)
            class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
            Id = class_dict[ann['class']]
            # print(Id)
            csvwriter.writerow([ann['width'],ann['height'],ann['xmin'],ann['ymin'],ann['xmax'],ann['ymax'],Id,ann['filename']])
            # print(ann['filename'], ann['class'])

    return pd.DataFrame(ann_list)
def get_test_df(ann_path, img_path):
    ann_path_list = get_file_list(ann_path, '.xml')
    ann_list = []
    with open('Test.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['Width', 'Height', 'Roi.X1' , 'Roi.Y1' , 'Roi.X2' , 'Roi.Y2' , 'ClassId' , 'Path'])
        for a_path in ann_path_list:
            root = ET.parse(a_path).getroot()
            ann = {}
            ann['filename'] = Path(str(img_path) + '/' + root.find("./filename").text)
            ann['width'] = root.find("./size/width").text
            ann['height'] = root.find("./size/height").text
            ann['class'] = root.find("./object/name").text
            ann['xmin'] = int(root.find("./object/bndbox/xmin").text)
            ann['ymin'] = int(root.find("./object/bndbox/ymin").text)
            ann['xmax'] = int(root.find("./object/bndbox/xmax").text)
            ann['ymax'] = int(root.find("./object/bndbox/ymax").text)
            ann_list.append(ann)
            class_dict = {'speedlimit': 0, 'stop': 1, 'crosswalk': 2, 'trafficlight': 3}
            Id = class_dict[ann['class']]
            # print(Id)
            csvwriter.writerow([ann['width'],ann['height'],ann['xmin'],ann['ymin'],ann['xmax'],ann['ymax'],Id,ann['filename']])
            # print(ann['filename'], ann['class'])

    return pd.DataFrame(ann_list)


base_path = 'train/'
ann_path = base_path + 'annotations/'
img_path = base_path + 'images/'
test_base_path = 'test/'
test_ann_path = test_base_path + 'annotations/'
test_img_path = test_base_path + 'images/'
"""
Uzywane tylko podczas tworzenia plikow csv :
"""
# df_train = get_train_df(ann_path, img_path)
# df_test = get_test_df(test_ann_path,test_img_path)



