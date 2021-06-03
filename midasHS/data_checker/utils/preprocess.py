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


def hs_dataset(transformations=[], generator='data_generator_for_npy'):
    generators = {
    'data_generator_for_gifs': (data_generator_for_gifs, (tf.float32, tf.float32)),
    'data_generator': (data_generator, (tf.float32, tf.float32)),
    'data_generator_only_normal': (data_generator_only_normal, (tf.float32)),
    'data_generator_only_golden': (data_generator_only_golden, (tf.float32)),
    'data_generator_for_npy': (data_generator_for_npy, (tf.float32, tf.float32))
    }
    dataset = tf.data.Dataset.from_generator(generators.get(generator)[0], generators.get(generator)[1])
    for transform in transformations:
        # print(transform)
        dataset = dataset.map(transform)
    return dataset

def hs_keras_dataset(transformations=[], split_ratio=0.7, batch_size=32, x=True, y=True):
    data_dir = 'data_checker/utils/cropped_images/'
    dir = list(pathlib.Path(data_dir).glob('*/'))
    size = len(dir)
    split = int(size * split_ratio)
    if x and y:
        training_generator = HSKerasDataGenerator(dir[:split], transforms=transformations, batch_size=batch_size)
        testing_generator = HSKerasDataGenerator(dir[split:], transforms=transformations, batch_size=batch_size)
    if x:
        training_generator = HSKerasDataGeneratorX(dir[:split], transforms=transformations, batch_size=batch_size)
        testing_generator = HSKerasDataGeneratorX(dir[split:], transforms=transformations, batch_size=batch_size)
    return training_generator, testing_generator

def dataset_size():
    data_dir = 'data_checker/utils/cropped_images/'
    return len(list(pathlib.Path(data_dir).glob('*/')))

def data_generator():
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        golden = [str(g) for g in list(pathlib.Path(curr_image_dir).glob('./golden/*.png'))]
        normal = str(list(pathlib.Path(curr_image_dir).glob('./normal/*.png'))[0])

        golden = sorted(golden, key=get_img_number)

        normal_image_tensor = png_path_to_tensor(normal)
        golden_image_tensor = tf.stack([png_path_to_tensor(img) for img in golden])

        yield normal_image_tensor, golden_image_tensor


def data_generator_for_gifs():
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'
        golden = str(curr_image_dir) + '/' + name + '_golden_cropped.gif'
        normal_image_tensor = png_path_to_tensor(normal)
        golden_image_tensor = gif_path_to_tensor(golden)
        yield normal_image_tensor, golden_image_tensor

def data_generator_only_normal():
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'
        normal_image_tensor = png_path_to_tensor(normal)
        yield normal_image_tensor

def data_generator_only_golden():
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        golden = str(curr_image_dir) + '/' + name + '_golden_cropped.gif'
        golden_image_tensor = gif_path_to_tensor(golden)
        yield golden_image_tensor

def data_generator_for_npy():
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        normal = str(curr_image_dir) + '/' + name + '_normal.npy'
        golden = str(curr_image_dir) + '/' + name + '_golden.npy'
        normal_image_tensor = np.load(normal)
        golden_image_tensor = np.load(golden)
        yield normal_image_tensor, golden_image_tensor


def png_path_to_tensor(path, channels=3):
    image = tf.io.read_file(path)
    image_tensor = tf.image.decode_png(image, channels=channels)
    return image_tensor

def gif_path_to_tensor(path):
    image = tf.io.read_file(path)
    image_tensor = tf.io.decode_gif(image)
    return image_tensor

def dir_to_gif(dir):
    pass

def get_img_number(path, note="golden_cropped_"):
    name = os.path.basename(path)
    index = name.find(note)
    num = int(name[index+len(note):-4])
    # print(num)
    return num

def save_data_as_npy():
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'
        golden = str(curr_image_dir) + '/' + name + '_golden_cropped.gif'
        normal_image_tensor = png_path_to_tensor(normal)
        np.save(str(curr_image_dir) + '/' + name + '_normal.npy', normal_image_tensor)
        print('saved {} normal'.format(name))
        golden_image_tensor = gif_path_to_tensor(golden)
        np.save(str(curr_image_dir) + '/' + name + '_golden.npy', golden_image_tensor)
        print('saved {} golden'.format(name))

class HSKerasDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dir, batch_size=32, shuffle=True, transforms=[]):
        self.x_dim = (200,200,3)
        self.y_dim = (100,200,200,3)
        self.batch_size = batch_size
        self.file_dirs = dir
        self.shuffle = shuffle
        self.transforms = transforms
        self.on_epoch_end()

    def __getitem__(self, index):
        # generate a batch of data
        dir = self.file_dirs[index*self.batch_size:(index+1)*self.batch_size]
        x, y = self.__data_generation(dir)
        return x, y
        # return *self.__data_generation(dir),

    def __data_generation(self, dirs_list):
        x = np.empty((self.batch_size, *self.x_dim))

        y = np.empty((self.batch_size, *self.y_dim))

        for i, curr_image_dir in enumerate(dirs_list):
            name = str(curr_image_dir).split('\\')[-1]
            normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'

            x_tensor = png_path_to_tensor(normal)
            for t in self.transforms:
                x_tensor = t(x_tensor)
            x[i,] = x_tensor

            golden = str(curr_image_dir) + '/' + name + '_golden_cropped.gif'
            y_tensor = gif_path_to_tensor(golden)
            for t in self.transforms:
                y_tensor = t(y_tensor)
            y[i,] = y_tensor
        return x, y

    def __len__(self):
        return len(self.file_dirs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_dirs)

class HSKerasDataGeneratorX(tf.keras.utils.Sequence):
    def __init__(self, dir, batch_size=32, shuffle=True, transforms=[]):
        self.x_dim = (200,200,3)
        # self.y_dim = (100,200,200,3)
        self.batch_size = batch_size
        self.file_dirs = dir
        self.shuffle = shuffle
        self.transforms = transforms
        self.on_epoch_end()

    def __getitem__(self, index):
        # generate a batch of data
        dir = self.file_dirs[index*self.batch_size:(index+1)*self.batch_size]
        x = self.__data_generation(dir)
        return x
        # return *self.__data_generation(dir),

    def __data_generation(self, dirs_list):
        x = np.empty((self.batch_size, *self.x_dim))

        # y = np.empty((self.batch_size, *self.y_dim))

        for i, curr_image_dir in enumerate(dirs_list):
            name = str(curr_image_dir).split('\\')[-1]
            normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'

            x_tensor = png_path_to_tensor(normal)
            for t in self.transforms:
                x_tensor = t(x_tensor)
            x[i,] = x_tensor

            # golden = str(curr_image_dir) + '/' + name + '_golden_cropped.gif'
            # y_tensor = gif_path_to_tensor(golden)
            # for t in self.transforms:
            #     y_tensor = t(y_tensor)
            # y[i,] = y_tensor
        return x

    def __len__(self):
        return len(self.file_dirs)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_dirs)


if __name__ == "__main__":
    d = hs_dataset()
    # save(d, "cropped_dataset_1.tfrecord")
    # print(d)
