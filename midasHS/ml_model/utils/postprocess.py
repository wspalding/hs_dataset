import tensorflow as tf
from PIL import Image
import os
import shutil

def reScale_0_1_255(x):
    return x*255

def reScale_n1_1_255(x):
    return (x * 127.5) + 127.5

def convert_to_png(tensor, save_file='example.png', path='ml_model/static/'):
    tensor = reScale_n1_1_255(tensor)
    tensor = tf.cast(tensor, tf.uint8)
    img = tf.image.encode_png(tensor)
    tf.io.write_file(path + save_file, img)
    return save_file

def convert_to_gif(tensor, save_file='example.gif', path='ml_model/static/'):
    temp_dir = path + 'temp/'
    os.makedirs(temp_dir, exist_ok=True)
    frames = [convert_to_png(tensor[i], save_file='temp_{}.png'.format(i), path=temp_dir) for i in range(len(tensor))]
    frames = [Image.open(temp_dir + f) for f in frames]
    frames[0].save(path+save_file, format='GIF', append_images=frames[1:], save_all=True, duration=len(tensor)//2, loop=0)
    # shutil.rmtree(temp_dir)
    return save_file
