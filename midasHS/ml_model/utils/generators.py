import tensorflow as tf
import numpy as np
from data_checker.utils.preprocess import png_path_to_tensor, gif_path_to_tensor
from data_checker.utils.preprocess import data_generator_only_normal
from tensorflow.keras.utils import to_categorical

# class HSKerasMixedDataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, dir, generator, seed_gen_dim, batch_size=32, shuffle=True, transforms=[], noise=0.3):
#         self.x_dim = (200,200,3)
#         self.y_dim = (2)
#         self.seed_gen_dim = seed_gen_dim
#         self.generator = generator
#         self.batch_size = batch_size
#         self.file_dirs = dir
#         self.shuffle = shuffle
#         self.transforms = transforms
#         self.noise = noise
#         self.on_epoch_end()
#
#     def __getitem__(self, index):
#         # generate a batch of data
#         dir = self.file_dirs[index*self.batch_size:min((index+1)*self.batch_size, len(self.file_dirs))]
#         x, y = self.__data_generation(dir)
#         return x, y
#         # return *self.__data_generation(dir),
#
#     def __data_generation(self, dirs_list):
#         data = np.empty((self.batch_size, *self.x_dim))
#
#         # y = np.empty((self.batch_size, *self.y_dim))
#
#         for i, curr_image_dir in enumerate(dirs_list):
#             name = str(curr_image_dir).split('\\')[-1]
#             normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'
#
#             x_tensor = png_path_to_tensor(normal)
#             for t in self.transforms:
#                 x_tensor = t(x_tensor)
#             data[i,] = x_tensor
#
#         seeds = np.random.normal(0, 1, (data.shape[0], self.seed_gen_dim))
#         fake_train_batch = self.generator.predict(seeds)
#         # print(data.shape, fake_train_batch.shape)
#
#         combined = np.concatenate([data, fake_train_batch])
#
#         labels = np.zeros(combined.shape[0])
#         labels[:data.shape[0]] = 1
#
#         indices = np.arange(combined.shape[0])
#         np.random.shuffle(indices)
#         combined = combined[indices]
#         labels = labels[indices]
#         # combined.shape += (1,)
#
#         labels = to_categorical(labels)
#
#         add_noise(labels, self.noise)
#         # print(combined.shape, labels.shape)
#
#         return combined, labels
#
#     def __len__(self):
#         return (len(self.file_dirs)*2 // self.batch_size)
#
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.file_dirs)





"""
Augmented version of the mixed data generator that includes classes to as an extra input
"""
class HSKerasMixedDataGeneratorWithClasses(tf.keras.utils.Sequence):
    def __init__(self, dir, generator, seed_gen_dim, batch_size=32, shuffle=True, transforms=[], noise=0.2):
        self.x_dim = (200,200,3)
        self.num_classes = 6
        self.y_dim = (2)
        self.seed_gen_dim = seed_gen_dim
        self.generator = generator
        self.batch_size = batch_size
        self.file_dirs = dir
        self.shuffle = shuffle
        self.transforms = transforms
        self.noise = noise
        self.on_epoch_end()

    def __getitem__(self, index):
        # generate a batch of data
        dir = self.file_dirs[index*self.batch_size:min((index+1)*self.batch_size, len(self.file_dirs))]
        x, y = self.__data_generation(dir)
        return x, y
        # return *self.__data_generation(dir),

    def __data_generation(self, dirs_list):
        data = np.empty((self.batch_size, *self.x_dim))
        clss = np.empty((self.batch_size, 1))
        # y = np.empty((self.batch_size, *self.y_dim))
        for i, curr_image_dir in enumerate(dirs_list):
            name = str(curr_image_dir).split('\\')[-1]
            sections = name.split('_')
            type = self.type_to_class(sections[-1])
            name = '_'.join(sections[:-1])
            normal = str(curr_image_dir) + '/' + name + '_normal_cropped.png'

            x_tensor = png_path_to_tensor(normal)
            for t in self.transforms:
                x_tensor = t(x_tensor)
            clss[i,] = type
            data[i,] = x_tensor

        seeds = np.random.normal(0, 1, (data.shape[0], self.seed_gen_dim))

        rand_classes = np.random.randint(0,6, size=(self.batch_size, 1))
        combined_clsses = np.concatenate([clss, rand_classes])

        fake_train_batch = self.generator.predict([seeds, rand_classes])
        combined = np.concatenate([data, fake_train_batch])

        labels = np.zeros(combined.shape[0])
        labels[:data.shape[0]] = 1

        self.add_label_noise(labels, diff=self.noise)

        self.add_img_noise(combined)

        indices = np.arange(combined.shape[0])
        np.random.shuffle(indices)
        combined = combined[indices]
        combined_clsses = combined_clsses[indices]
        labels = labels[indices]
        # combined.shape += (1,)

        # labels = to_categorical(labels)
        # add_noise(labels, self.noise)
        #

        # print(combined.shape, labels.shape)

        return [combined, combined_clsses], labels

    def __len__(self):
        return (len(self.file_dirs)*2 // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_dirs)

    def type_to_class(self, type):
        if type == 'Minion':
            return 0
        elif type == 'Hero Power':
            return 1
        elif type == 'Hero':
            return 2
        elif type == 'Playable Hero':
            return 3
        elif type == 'Weapon':
            return 4
        else: # type == 'Ability'
            return 5

    def add_img_noise(self, imgs, mean=0, var=0.15):
        sigma = var**5
        img_shape = imgs[0].shape
        for i in range(len(imgs)):
            gaussian_noise = np.random.normal(mean, sigma, img_shape)
            imgs[i] += gaussian_noise

    def add_label_noise(self, labels, diff=0.1):
        for i in range(len(labels)):
            noise = np.random.uniform(0.0, diff)
            if labels[i] < 0.5:
                labels[i] += noise
            else:
                labels[i] -= noise
        # return labels






def add_noise(labels, high=0.3):
    # print(labels.shape)
    for label in labels:
        noise = np.random.uniform(0.0,high)
        if label[0] == 0.0:
            label[0] += noise
            label[1] -= noise
        else:
            label[0] -= noise
            label[1] += noise

        if np.random.uniform(0,1) > 0.05:
            tmp = label[0]
            label[0] = label[1]
            label[1] = tmp
