import wandb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Conv2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D,LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras import initializers
from tensorflow.keras.metrics import TruePositives, FalseNegatives, FalsePositives, TrueNegatives
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LambdaCallback
from ml_model.utils.generators import HSKerasMixedDataGeneratorWithClasses
from ml_model.utils.models import create_descriminator, create_generator, create_joint_model, GEN_LR, DESC_LR

from ml_model.utils.metrics import BinaryTruePositiveWithNoise, BinaryTrueNegativeWithNoise, BinaryFalsePositiveWithNoise, BinaryFalseNegativeWithNoise


from tensorflow.keras.models import load_model

import numpy as np
import os
import pathlib


def init_wandb(project='midashs', sync_tensorboard=True, **kwargs):
    if kwargs.get('reconnect', False):
        wandb.init(project=project, sync_tensorboard=sync_tensorboard, resume=True)
    else:
        id = wandb.util.generate_id()
        wandb.init(project=project, id=id, sync_tensorboard=sync_tensorboard, resume="allow")
    config = wandb.config
    # params = {
    #     'adversarial_epochs'
    # }
    # config.update(params, allow_val_change=True)
    config.adversarial_epochs = kwargs.get('adversarial_epochs', 1000)
    config.descriminator_epochs = kwargs.get('descriminator_epochs', 2)
    config.descriminator_examples = kwargs.get('descriminator_examples', 10000)
    config.generator_epochs = kwargs.get('generator_epochs',12)
    config.generator_examples = kwargs.get('generator_examples', 10000)
    config.generator_seed_dim = kwargs.get('generator_seed_dim', 10)
    config.generator_conv_size = kwargs.get('generator_conv_size',64)
    config.batch_size = kwargs.get('batch_size', 100)
    config.image_shape = kwargs.get('image_shape', (200,200,3))
    config.noise = kwargs.get('noise', 0.3)
    config.num_classes = kwargs.get('num_classes', 6)
    return config


def log_descriminator(epochs, logs):
    if logs.get('acc', 0) == 0:
        acc = (logs.get('DescTP')+logs.get('DescTN'))/(logs.get('DescTP') + logs.get('DescFN')+logs.get('DescTN') + logs.get('DescFP'))
    else:
        acc = logs.get('acc')

    if logs.get('DescTP', 0) == 0:
        sensitivity = 0
    else:
        sensitivity = logs.get('DescTP')/(logs.get('DescTP') + logs.get('DescFN'))

    if logs.get('DescTN', 0) == 0:
        specificity = 0
    else:
        specificity = logs.get('DescTN')/(logs.get('DescTN') + logs.get('DescFP'))

    if logs.get('val_acc', 0) == 0:
        val_acc = (logs.get('val_DescTP')+logs.get('val_DescTN'))/(logs.get('val_DescTP') + logs.get('val_DescFN')+logs.get('val_DescTN') + logs.get('val_DescFP'))
    else:
        val_acc = logs.get('val_acc')

    if logs.get('val_DescTP', 0) == 0:
        val_sensitivity = 0
    else:
        val_sensitivity = logs.get('val_DescTP')/(logs.get('val_DescTP') + logs.get('val_DescFN'))

    if logs.get('val_DescTN', 0) == 0:
        val_specificity = 0
    else:
        val_specificity = logs.get('val_DescTN')/(logs.get('val_DescTN') + logs.get('val_DescFP'))


    wandb.log({

        'descriminator_loss': logs.get('loss'),
        'descriminator_acc': acc,
        'descriminator_val_loss': logs.get('val_loss'),
        'descriminator_val_acc':val_acc,

        'desc_correct_label_rate_(sensitivity)': sensitivity,
        'desc_correct_neg_rate_(specificity)': specificity,

        'desc_val_correct_label_rate_(sensitivity)': val_sensitivity,
        'desc_val_correct_neg_rate_(specificity)': val_specificity,
        })


def generator_inputs(num_examples, config, **kwargs):
    classes = kwargs.get('classes', np.random.randint(0,6, size=(num_examples, 1)))
    return [np.random.normal(0,1, (num_examples, config.generator_seed_dim)), classes]

def train_descriminator(generator, desciminator, x_train, x_test, iter, config, save_file):

    desciminator.trainable = True
    generator.trainable = False

    train_generator = HSKerasMixedDataGeneratorWithClasses(x_train.file_dirs, generator, config.generator_seed_dim, batch_size=x_train.batch_size//2, transforms=x_train.transforms)
    testing_generator = HSKerasMixedDataGeneratorWithClasses(x_test.file_dirs, generator, config.generator_seed_dim, batch_size=x_test.batch_size//2, transforms=x_test.transforms)

    # desciminator.summary()

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_descriminator)

    history = desciminator.fit(train_generator,
                                epochs=config.descriminator_epochs,
                                # batch_size=config.batch_size,
                                validation_data=testing_generator,
                                callbacks=[wandb_logging_callback],
                                # workers=3,
                                # use_multiprocessing=True
                                )

    save_path = os.path.join(pathlib.Path().absolute(), 'ml_model/models/', save_file)
    desciminator.save(save_path)
    print('saving descriminator', save_path, os.path.exists(save_path))

def log_generator(epochs, logs):
    if logs.get('GenTP', 0) == 0:
        sensitivity = 0
    else:
        sensitivity = logs.get('GenTP')/(logs.get('GenTP')+logs.get('GenFN'))

    wandb.log({'generator_loss': logs.get('loss'),
                'generator_acc': logs.get('acc'),

                'gen_pos_rate_(sensitivity)': sensitivity,

                # 'generator_TN': logs.get('GenTN'),
                # 'generator_FN': logs.get('GenFN'),
                # 'generator_FP': logs.get('GenFP')
                })

def train_generator(generator, descriminator, joint_model, config, save_file):
    num_examples = config.generator_examples

    train = generator_inputs(num_examples, config)
    labels = np.ones(num_examples)
    #
    # labels = to_categorical(labels)
    # print(labels)
    generator.trainable = True
    descriminator.trainable = False

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)

    joint_model.fit(x=train, y=labels, epochs=config.generator_epochs,
                                        batch_size=config.batch_size,
                                        callbacks=[wandb_logging_callback],
                                        workers=10,
                                        use_multiprocessing=True)


    save_path = os.path.join(pathlib.Path().absolute(), 'ml_model/models/', save_file)
    generator.save(save_path)
    print('saving generator', save_path, os.path.exists(save_path))

# need to change this to use images not noise
def sample_images(generator, config, log=False):
    classes = [0,1,2,3,4,5,0,3,4,5]
    noise = generator_inputs(10, config, classes=np.asarray(classes))
    gen_imgs = generator.predict(noise)
    if log:
        log_samples(gen_imgs)
    return gen_imgs

def log_samples(gen_imgs):
    gen_imgs = (gen_imgs * 127.5) + 127.5
    print("samples:" , gen_imgs.shape)
    wandb.log({'examples': [wandb.Image(i) for i in gen_imgs]})


def train(dataset, dataset_size, config=None, generator_savefile=None, descriminator_savefile=None, save_models=None):
    if not config:
        config = init_wandb(batch_size=10)

    training_set, testing_set = dataset

    x_train = training_set
    x_test = testing_set

    try:
        print("loading generator")
        generator = load_model(os.path.join(pathlib.Path().absolute(), 'ml_model/models/', generator_savefile))
    except OSError:
        print("creating generator")
        generator = create_generator(config)
    generator.summary()


    try:
        print("loading descriminator")
        desc_file = os.path.join(pathlib.Path().absolute(), 'ml_model/models/', descriminator_savefile)
        descriminator = load_model(desc_file, compile=False)

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
        descriminator.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    except OSError:
        print("creating descriminator")
        descriminator = create_descriminator(config)
    descriminator.summary()

    joint_model = create_joint_model(generator, descriminator)

    samples = []

    # print("training generator")
    # train_generator(generator, descriminator, joint_model, config, generator_savefile)

    for i in range(config.adversarial_epochs):
        print("adversarial epoch: {}/{}".format(i, config.adversarial_epochs))

        print("training descriminator")
        train_descriminator(generator, descriminator, x_train, x_test, i, config, descriminator_savefile)

        print("training generator")
        train_generator(generator, descriminator, joint_model, config, generator_savefile)

        print("generating samples")
        # if i % 3 == 0:
        #     sample_images(generator, config, log=True)
        sample_images(generator, config, log=True)

    return sample_images(generator, config)
