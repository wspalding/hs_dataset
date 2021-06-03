import tensorflow as tf
from tensorflow.keras.metrics import Metric as kerasMetric
import tensorflow.keras.backend as K


class CatagoricalTruePositive(kerasMetric):
    def __init__(self, name='cat_TP', **kwargs):
        super(CatagoricalTruePositive, self). __init__(name=name, **kwargs)

        # self.batch_size = batch_size
        # self.num_classes = num_classes

        self.cat_true_pos = self.add_weight(name='ctp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)

        true_pos = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))

        self.cat_true_pos.assign_add(true_pos)

    def result(self):
        return self.cat_true_pos


class BinaryTruePositiveWithNoise(kerasMetric):
    def __init__(self, name='BTPwN', **kwargs):
        super(BinaryTruePositiveWithNoise, self). __init__(name=name, **kwargs)
        self.cat_true_pos = self.add_weight(name='ctp', initializer='zeros')
        self.threshold = kwargs.get('threshold', 0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = K.greater(y_true, self.threshold)
        y_pred = K.greater(y_pred, self.threshold)
        pos = tf.math.logical_and(y_true, y_pred)
        # print(pos)
        # print(K.equal(y_true, y_pred))
        true_pos = K.sum(K.cast(pos, dtype=tf.float32))
        self.cat_true_pos.assign_add(true_pos)

    # def get_config(self):
    #     config = {}
    #     base_config = super(BinaryTruePositiveWithNoise, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    def result(self):
        return self.cat_true_pos

    def __str__(self):
        return 'BinaryTruePositiveWithNoise'





class BinaryTrueNegativeWithNoise(kerasMetric):
    def __init__(self, name='BTNwN', **kwargs):
        super(BinaryTrueNegativeWithNoise, self). __init__(name=name, **kwargs)
        self.cat_true_neg = self.add_weight(name='ctp', initializer='zeros')
        self.threshold = kwargs.get('threshold', 0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = K.less(y_true, self.threshold)
        y_pred = K.less(y_pred, self.threshold)
        pos = tf.math.logical_and(y_true, y_pred)
        # print(pos)
        # print(K.equal(y_true, y_pred))
        true_pos = K.sum(K.cast(pos, dtype=tf.float32))
        self.cat_true_neg.assign_add(true_pos)

    # def get_config(self):
    #     config = {}
    #     base_config = super(BinaryTruePositiveWithNoise, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    def result(self):
        return self.cat_true_neg

    def __str__(self):
        return 'BinaryTrueNegativeWithNoise'


class BinaryFalsePositiveWithNoise(kerasMetric):
    def __init__(self, name='BFPwN', **kwargs):
        super(BinaryFalsePositiveWithNoise, self). __init__(name=name, **kwargs)
        self.cat_flase_neg = self.add_weight(name='ctp', initializer='zeros')
        self.threshold = kwargs.get('threshold', 0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = K.less(y_true, self.threshold)
        y_pred = K.greater(y_pred, self.threshold)
        pos = tf.math.logical_and(y_true, y_pred)
        # print(pos)
        # print(K.equal(y_true, y_pred))
        true_pos = K.sum(K.cast(pos, dtype=tf.float32))
        self.cat_flase_neg.assign_add(true_pos)

    # def get_config(self):
    #     config = {}
    #     base_config = super(BinaryTruePositiveWithNoise, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    def result(self):
        return self.cat_flase_neg

    def __str__(self):
        return 'BinaryFalsePositiveWithNoise'


class BinaryFalseNegativeWithNoise(kerasMetric):
    def __init__(self, name='BFNwN', **kwargs):
        super(BinaryFalseNegativeWithNoise, self). __init__(name=name, **kwargs)
        self.cat_false_neg = self.add_weight(name='ctp', initializer='zeros')
        self.threshold = kwargs.get('threshold', 0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = K.greater(y_true, self.threshold)
        y_pred = K.less(y_pred, self.threshold)
        pos = tf.math.logical_and(y_true, y_pred)
        # print(pos)
        # print(K.equal(y_true, y_pred))
        true_pos = K.sum(K.cast(pos, dtype=tf.float32))
        self.cat_false_neg.assign_add(true_pos)

    # def get_config(self):
    #     config = {}
    #     base_config = super(BinaryTruePositiveWithNoise, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    def result(self):
        return self.cat_false_neg

    def __str__(self):
        return 'BinaryFalseNegativeWithNoise'
