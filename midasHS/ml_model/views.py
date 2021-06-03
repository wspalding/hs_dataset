from django.shortcuts import render, redirect
from django.http import JsonResponse
from data_checker.utils.preprocess import hs_dataset, dataset_size, hs_keras_dataset
from ml_model.utils.transforms import *
from ml_model.utils.train import *
from ml_model.utils.generators import *
from ml_model.utils.postprocess import convert_to_png, convert_to_gif
from tensorflow.keras.models import load_model
from ml_model.utils.metrics import BinaryTruePositiveWithNoise

import time

import tensorflow as tf

num = '0000'
GEN_DIM_SEED = 30
# Create your views here.
def index(request, **kwargs):
    # dataset = hs_dataset(transformations=[normalize])
    # num = '0002'
    gen_file = kwargs.get('gen_file', "generator_test_{}.h5".format(num))
    desc_file = kwargs.get('desc_file', "descriminator_test_{}.h5".format(num))
    custom_objects = {'BinaryTruePositiveWithNoise': BinaryTruePositiveWithNoise()}
    batch_size = 32
    gen = os.path.join(pathlib.Path().absolute(), 'ml_model/models/', gen_file)
    des = os.path.join(pathlib.Path().absolute(), 'ml_model/models/', desc_file)
    generator = load_model(gen, custom_objects=custom_objects)
    desciminator = load_model(des, custom_objects=custom_objects, compile=False)
    data_dir = 'data_checker/utils/cropped_images/'
    dir = list(pathlib.Path(data_dir).glob('*/'))
    training_set = HSKerasMixedDataGeneratorWithClasses(dir, generator, GEN_DIM_SEED, transforms=[normalize_negative1_1])
    # training_set, testing_set = split_dataset(dataset, dataset_size())

    # elements = list(dataset.take(100).as_numpy_iterator())
    # print(len(elements))
    # display_cards = [2,4,5,7,14,22,54,55,76,87,99]
    x_batch, y_batch = training_set.__getitem__(0)
    model_preds = desciminator.predict(x_batch)
    # print(x_batch.shape, y_batch.shape)
    display_images = []
    for i in range(batch_size):
        x = x_batch[0][i]
        clss = x_batch[1][i]
        # print(x.shape, clss.shape)
        y = y_batch[i]
        normal_img = convert_to_png(x, save_file='example_{}.png'.format(i))
        gold_img = None #convert_to_gif(y, save_file='example_{}.gif'.format(i))
        model_pred = model_preds[i]
        # print(model_pred.shape)
        display_images.append({'number': i,
                                'x_tensor': x,
                                'normal_img': normal_img,
                                'y_tensor': y,
                                'class': clss,
                                'gold_img': gold_img,
                                'model_pred': model_pred})

    context = {'display_images': display_images}
    return render(request, 'model_index.html', context)


def train_model(request):
    # dataset = hs_dataset(transformations=[normalize1], generator='data_generator_only_normal')
    # run_name = 'test_3_full_run'
    # num = '0002'
    generator_savefile = "generator_test_{}.h5".format(num)
    descriminator_savefile = "descriminator_test_{}.h5".format(num)
    config = init_wandb(reconnect=False,
                        generator_examples=5000,
                        descriminator_examples=dataset_size(),
                        batch_size=10,
                        adversarial_epochs=1000,
                        descriminator_epochs=1,
                        generator_epochs=1,
                        generator_seed_dim=GEN_DIM_SEED,
                        noise=0.1)
    training_set, testing_set = hs_keras_dataset(transformations=[normalize_negative1_1],
                                                    batch_size=config.batch_size,
                                                    y=False)
    samples = train((training_set, testing_set), dataset_size(), config=config,
                                                                    generator_savefile=generator_savefile,
                                                                    descriminator_savefile=descriminator_savefile,
                                                                    save_models=False)
    display_images = []
    for i, sample in enumerate(samples):
        img = convert_to_png(sample, save_file='example_{}.png'.format(i))
        display_images.append([i, img, '', ''])
    context = {'display_images': display_images}
    return redirect('/ml_model', gen_file=generator_savefile, desc_file=descriminator_savefile)


def time_function(request):
    start = time.time()
    dataset = hs_dataset(transformations=[normalize2],
                            generator='data_generator_for_gifs')
    i = 0
    for _ in dataset:
        i += 1
    end = time.time()
    t = end - start
    return JsonResponse({"time elapsed":"{} seconds".format(t),"count":i})
