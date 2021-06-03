import tensorflow as tf


def normalize2(x,y):
    return x/255, y/255

def normalize1(x):
    return x/255

def normalize_negative1_1(x):
    x = tf.cast(x, tf.float32)
    return (x - 127.5) / 127.5
