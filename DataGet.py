import os
import pickle
from numpy import array, zeros

from PIL import Image

def jpg2array(in_dir:str, out_file:str, jpg_num:int, width=224, height=224, channals=3):
    label_list = os.listdir(in_dir)
    data = zeros((jpg_num, width, height, channals), dtype = 'uint8')
    label_dic = {"left":0, "right":1, "up":2, "down":3, "leftright":4,
                 "updown":5, "true":6, "false":7}
    labels = zeros(jpg_num)
    i = 0
    for label in label_list:
        jpg_dir = os.path.join(in_dir, label)
        jpg_list = os.listdir(jpg_dir)
        for jpg_file in jpg_list:
            jpg = os.path.join(jpg_dir, jpg_file)
            img = Image.open(jpg)
            data[i] = array(img)
            labels[i] = label_dic[label]
            i+=1
    pickle.dump((data, labels), open(out_file, 'wb'))

def load_data(file_dir):
    data, label = pickle.load(open(file_dir, 'rb'))
    return data, label