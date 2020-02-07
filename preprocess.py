import pandas as pd
# import urllib
import os
import pathlib
# tensorflow stuff
import tensorflow as tf

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def hs_dataset():
    data_dir = 'cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))

    # curr_image_dir = image_dirs[0]
    normal_cards = []
    golden_cards = []
    card_number = 0
    total = len(image_dirs)
    for curr_image_dir in image_dirs:
        # card_dir = list(pathlib.Path(curr_image_dir).glob('*/'))

        print("\r\r {}/{}: {}  ({:.4f}% complete)".format(card_number, total, str(curr_image_dir), (card_number/total*100)), end='\r\r')

        golden = [str(g) for g in list(pathlib.Path(curr_image_dir).glob('./golden/*.png'))]
        normal = str(list(pathlib.Path(curr_image_dir).glob('./normal/*.png'))[0])

        golden = sorted(golden, key=get_img_number)

        normal_image_tensor = png_path_to_tensor(normal)
        golden_image_tensor = tf.stack([png_path_to_tensor(img) for img in golden])

        normal_cards.append(normal_image_tensor)
        golden_cards.append(golden_image_tensor)
        # print(normal_image_tensor.shape)
        # print(golden_image_tensor.shape)
        card_number += 1

    dataset = tf.data.Dataset.from_tensor_slices((normal_cards, golden_cards))
    return dataset


def png_path_to_tensor(path, channels=3):
    image = tf.io.read_file(path)
    image_tensor = tf.image.decode_png(image, channels=channels)
    return image_tensor

def get_img_number(path, note="golden_cropped_"):
    name = os.path.basename(path)
    index = name.find(note)
    num = int(name[index+len(note):-4])
    # print(num)
    return num

def save(dataset, file_name):
    print("saving dataset in file: {}".format(filename))
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset)

if __name__ == "__main__":
    d = hs_dataset()
    save(d, "cropped_dataset_1.tfrecord")
    # print(d)
