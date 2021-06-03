import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, Conv2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D, UpSampling3D,LeakyReLU, Dropout, \
    BatchNormalization, MaxPool2D, Input, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import FalsePositives, FalseNegatives, TruePositives, TrueNegatives, CategoricalAccuracy
from tensorflow.keras.metrics import Precision, Recall
# from sklearn.metrics import classification_report

from ml_model.utils.metrics import BinaryTruePositiveWithNoise, BinaryTrueNegativeWithNoise, BinaryFalsePositiveWithNoise, BinaryFalseNegativeWithNoise

# TODO:
#   fix scaling from 0-255 -> 0-1 -> -1-1
#
#

GEN_LR = 0.0001
DESC_LR = 0.0002
# LBL_SMOOTHING = 0.0


def create_descriminator(config):
    # desciminator = Sequential(name='descriminator')
    #
    # desciminator.add(Conv2D(8, (16,16), strides=(1,1), padding='same', input_shape=config.image_shape)) #shape -> 200,200,3 -> 200,200,8
    # desciminator.add(Dropout(0.2))
    # desciminator.add(BatchNormalization())
    # desciminator.add(LeakyReLU(0.2))
    #
    # # desciminator.add(MaxPool2D(pool_size=(2,2))) #shape -> 200,200,8 -> 100,100,8
    # desciminator.add(Conv2D(32, (8,8), strides=(2,2), padding='same')) #shape -> 100,100,8 -> 100,100,32
    # desciminator.add(Dropout(0.2))
    # desciminator.add(BatchNormalization())
    # desciminator.add(LeakyReLU(0.2))
    #
    # # desciminator.add(MaxPool2D(pool_size=(2,2))) #shape -> 100,100,32 -> 50,50,32
    # desciminator.add(Conv2D(32, (4,4), strides=(2,2), padding='same'))
    # desciminator.add(Dropout(0.2))
    # desciminator.add(BatchNormalization())
    # desciminator.add(LeakyReLU(0.2))
    #
    # # desciminator.add(MaxPool2D(pool_size=(2,2))) #shape -> 50,50,32 -> 25,25,32
    # desciminator.add(Conv2D(32, (4,4), strides=(2,2), padding='same'))
    # desciminator.add(Dropout(0.2))
    # desciminator.add(BatchNormalization())
    # desciminator.add(LeakyReLU(0.2))
    #
    # desciminator.add(Flatten())
    # desciminator.add(Dense(64))
    # desciminator.add(LeakyReLU(0.2))
    # desciminator.add(Dense(64))
    # desciminator.add(LeakyReLU(0.2))
    # desciminator.add(Dense(64))
    # desciminator.add(LeakyReLU(0.2))
    # desciminator.add(Dense(2, activation='sigmoid'))
    # loss = CategoricalCrossentropy()


    clss_input = Input(shape=(1,))
    clss = Embedding(config.num_classes, 50)(clss_input)
    clss = Dense(config.image_shape[0]*config.image_shape[1])(clss)
    clss = LeakyReLU(0.2)(clss)
    clss = Reshape((config.image_shape[0],config.image_shape[1], 1))(clss)

    img_input = Input(shape=config.image_shape)

    merge = Concatenate()([img_input, clss])
    img_layer = Conv2D(8, (16,16), strides=(1,1), padding='same', input_shape=config.image_shape)(merge)
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    # img_layer = (MaxPool2D(pool_size=(2,2))) #shape -> 200,200,8 -> 100,100,8
    img_layer = Conv2D(32, (8,8), strides=(2,2), padding='same')(img_layer) #shape -> 100,100,8 -> 100,100,32
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    # img_layer = (MaxPool2D(pool_size=(2,2))) #shape -> 100,100,32 -> 50,50,32
    img_layer = Conv2D(32, (4,4), strides=(2,2), padding='same')(img_layer)
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    # img_layer = (MaxPool2D(pool_size=(2,2))) #shape -> 50,50,32 -> 25,25,32
    img_layer = Conv2D(32, (4,4), strides=(2,2), padding='same')(img_layer)
    img_layer = Dropout(0.2)(img_layer)
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Flatten()(img_layer)
    img_layer = Dense(64)(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)
    img_layer = Dense(64)(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)
    img_layer = Dense(64)(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    out_layer = Dense(1, activation='sigmoid')(img_layer)
    # out_layer = Dense(2, activation='softmax')(img_layer)

    desciminator = Model([img_input, clss_input], out_layer, name='Descriminator')

    # loss = CategoricalCrossentropy()
    loss = BinaryCrossentropy()
    optimizer = Adam(learning_rate=DESC_LR, beta_1=0.5)
    metrics=['acc', BinaryTruePositiveWithNoise(name='DescTP'),
                    BinaryTrueNegativeWithNoise(name='DescTN'),
                    BinaryFalsePositiveWithNoise(name='DescFP'),
                    BinaryFalseNegativeWithNoise(name='DescFN'),
                    # FalseNegatives(name='DescFN', thresholds=0.5),
                    # FalsePositives(name='DescFP', thresholds=0.5),
                    # TruePositives(name='DescTP', thresholds=0.5),
                    # TrueNegatives(name='DescTN', thresholds=0.5)
                    ]
    desciminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return desciminator

def create_generator(config):

    random_dim = config.generator_seed_dim
    # img shape = (200,200,3)
    # generator = Sequential(name='generator')
    # generator.add(Dense(256, input_dim=random_dim, kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Dense(512))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Dense(1024))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Dense(1875))
    # generator.add(LeakyReLU(0.2))
    # generator.add(Reshape((25,25, 3)))
    #
    # generator.add(Conv2DTranspose(16, (5,5), strides=(2,2), padding='same')) # shape -> 50,50,3
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Conv2DTranspose(16, (5,5), strides=(2,2), padding='same')) # shape -> 100, 100, 3
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Conv2DTranspose(16, (5,5), strides=(2,2), padding='same')) # shape -> 200, 200, 3
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Conv2DTranspose(16, (5,5), strides=(1,1), padding='same')) # shape -> 200, 200, 3
    # generator.add(BatchNormalization())
    # generator.add(LeakyReLU(0.2))
    #
    # generator.add(Conv2DTranspose(3, (3,3), strides=(1,1), padding='same', activation='tanh'))

    clss_input = Input(shape=(1,))
    clss = Embedding(config.num_classes, 50)(clss_input)
    clss = Dense(300)(clss)
    clss = Dense(25*25)(clss)
    clss = LeakyReLU(0.2)(clss)
    clss = Reshape((25,25, 1))(clss)
    clss = Conv2DTranspose(3, (5,5), strides=(1,1), padding='same')(clss)

    gen_seed_input = Input(shape=(random_dim,))

    seed_layer = Dense(256, input_dim=random_dim, kernel_initializer=tf.random_normal_initializer(stddev=0.02))(gen_seed_input)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Dense(512)(seed_layer)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Dense(1024)(seed_layer)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Dense(1875)(seed_layer)
    seed_layer = LeakyReLU(0.2)(seed_layer)
    seed_layer = Reshape((25,25, 3))(seed_layer)

    img_layer = Concatenate()([seed_layer, clss])

    img_layer = Conv2DTranspose(16, (10,10), strides=(2,2), padding='same')(img_layer) # shape -> 50,50,3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Conv2DTranspose(32, (7,7), strides=(2,2), padding='same')(img_layer) # shape -> 100, 100, 3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Conv2DTranspose(32, (5,5), strides=(2,2), padding='same')(img_layer) # shape -> 200, 200, 3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    img_layer = Conv2DTranspose(32, (5,5), strides=(1,1), padding='same')(img_layer) # shape -> 200, 200, 3
    img_layer = BatchNormalization()(img_layer)
    img_layer = LeakyReLU(0.2)(img_layer)

    out_layer = Conv2DTranspose(3, (3,3), strides=(1,1), padding='same', activation='tanh')(img_layer)


    generator = Model([gen_seed_input, clss_input], out_layer, name='Generator')

    return generator

def create_joint_model(generator, descriminator):
    # joint_model = Sequential()
    # joint_model.add(generator)
    # joint_model.add(descriminator)

    descriminator.trainable = False

    gen_seed_input, gen_clss_input = generator.input
    gen_output = generator.output

    gan_output = descriminator([gen_output, gen_clss_input])

    joint_model = Model([gen_seed_input, gen_clss_input], gan_output)
    optimizer = Adam(learning_rate=GEN_LR, beta_1=0.5)
    # loss = CategoricalCrossentropy()
    loss = BinaryCrossentropy()
    metrics = ['acc', BinaryTruePositiveWithNoise(name='GenTP'),
                        BinaryFalseNegativeWithNoise(name='GenFN'),
                        # FalseNegatives(name='GenFN', thresholds=0.5),
                        # TruePositives(name='GenTP', thresholds=0.5)
                        ]
    joint_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return joint_model
