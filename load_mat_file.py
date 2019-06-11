from PIL import Image
import csv
import pandas as pd
import scipy.io as spio
import cv2
import os

devkit = 'devkit'
class_name_file = os.path.join(devkit,"cars_meta.mat")
cars_train_list = os.path.join(devkit, "cars_train_annos.mat")
cars_test_list = os.path.join(devkit, "cars_test_annos.mat")

car_dataset = "car_dataset"
train_path = os.path.join(car_dataset, "cars_train")
test_path = os.path.join(car_dataset, "cars_test")

cars_meta = spio.loadmat(class_name_file)
cars_train_annos = spio.loadmat(cars_train_list)
cars_test_annos = spio.loadmat(cars_test_list)

labels = [c for c in cars_meta['class_names'][0]]
labels = pd.DataFrame(labels, columns=['labels'])

frame = [[i.flat[0] for i in line] for line in cars_train_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class', 'fname']
df_train = pd.DataFrame(frame, columns=columns)
df_train['class'] = df_train['class']-1
df_train['fname'] = [os.path.join(train_path,f) for f in df_train['fname']]

df_train = df_train.merge(labels, left_on='class', right_index=True)
df_train = df_train.sort_index()

frame = [[i.flat[0] for i in line] for line in cars_test_annos['annotations'][0]]
columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'fname']
df_test = pd.DataFrame(frame, columns=columns)
df_test['fname'] = [os.path.join(test_path,f) for f in df_test['fname']]

export_csv = df_train.to_csv (train_csv_file, index = None, header=True)
export_csv = df_test.to_csv (test_csv_file, index = None, header=True)
