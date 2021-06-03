from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse

from data_checker.utils.crop_images import get_points_from_file, crop_image_with_mask, processGif, create_mask, bounding_box, crop_image, get_file_number, get_crop_points
from data_checker.utils.preprocess import hs_dataset, png_path_to_tensor, get_img_number, save_data_as_npy
from data_checker.utils import misc
from data_checker.utils.download_cards import download_new_normal_cards_from_hearthpwn, download_new_golden_cards_from_hearthpwn
from data_checker.utils.image_manager import get_normal_img_src, get_golden_img_src, get_normal_img_path_from_id

import pandas as pd
import numpy as np
from PIL import Image, GifImagePlugin
import mechanicalsoup
import cv2
# import urllib
import requests
import shutil
import tempfile
import sys
import tensorflow as tf

from os import path, stat
import pathlib
import os
import re

from collections import defaultdict


# Create your views here.
def index(request):
    file = 'data_checker/utils/data_tracker.csv'
    if not os.path.exists(file):
        return redirect('data_checker/collect')
    data = pd.read_csv(file)
    context = {'data':[(row['card_id'], row['name'], row['normal_status'], row['golden_status'], row['img_url'], row['golden_img_url']) for index, row in data.iterrows()]}
    return render(request, 'checker.html', context)

def card(request, **kwargs):
    index = kwargs.get('card_index')
    # if request.method == 'POST':
    #     form = request.POST
    #     if 'Keep' in form:
    #         data.loc[index,'skip'] = False
    #         data.to_csv(file, index=False)
    #         return redirect('/data_checker/{}'.format(index + 1))
    #
    #     elif 'Skip' in form:
    #         data.loc[index,'skip'] = True
    #         data.to_csv(file, index=False)
    #         return redirect('/data_checker/{}'.format(index + 1))

    file = 'data_checker/utils/data_tracker.csv'
    data = pd.read_csv(file)
    context = {}

    row = data.loc[data['card_id'] == index]
    normal_img_src, normal_src_type = get_normal_img_src(row)
    golden_img_src, golden_src_type = get_golden_img_src(row)

    context['index'] = index
    context['normal'] = normal_img_src
    context['normal_type'] = normal_src_type
    context['normal_status'] = row.iloc[0]['normal_status']
    context['golden'] = golden_img_src
    context['golden_type'] = golden_src_type
    context['golden_status'] = row.iloc[0]['golden_status']
    return render(request, 'card.html', context)

def collect_new_cards(request):
    collect_new_cards_from_hearthpwn()
    return redirect('/data_checker/')


def download_normal_cards(request):
    print('downlaoding normal cards from hearthpwn')
    download_new_normal_cards_from_hearthpwn()
    return redirect('/data_checker/')

def download_golden_cards(request):
    print('downlaoding golden cards from hearthpwn')
    download_new_golden_cards_from_hearthpwn()
    return redirect('/data_checker/')

def label_normal_cards(request):
    file = 'data_checker/utils/data_tracker.csv'
    data = pd.read_csv(file)
    if request.method == "POST":
        for id, val in request.POST.items():
            if val == 'on':
                img_path = get_normal_img_path_from_id(id)
                if img_path[1] != 'path':
                    continue
                img_path = 'data_checker/static/' + img_path[0]
                type = data.loc[data['card_id'] == int(id) ,'card_type'].values[0]
                style = "normal"
                points = get_crop_points(type, style)
                print("cropping: ", img_path)
                crop_image(img_path, points, save_file=img_path)
                misc.add_dir_to_normal_dataset(id, type, data)
            else:
                print(id)
        data.to_csv(file, index=False)
        # return JsonResponse(request.POST)
    rows = data.loc[data['normal_status'] == 'downloaded'][:100]
    cards = []
    for i in range(len(rows)):
        id = rows.iloc[i]['card_id']
        src, type = get_normal_img_src(rows.iloc[[i]])
        card_type = rows.iloc[i]['card_type']
        status = rows.iloc[i]['normal_status']
        cards.append((id, src, type, card_type, status))
    context = {}
    context['title'] = 'normal'
    context['cards'] = cards
    return render(request, 'label.html', context)


def label_golden_cards(request):
    file = 'data_checker/utils/data_tracker.csv'
    data = pd.read_csv(file)
    rows = data.loc[data['golden_status'] == 'downloaded']
    cards = []
    for i in range(len(rows)):
        id = rows.iloc[i]['card_id']
        src, type = get_golden_img_src(rows.iloc[[i]])
        card_type = rows.iloc[i]['card_type']
        status = rows.iloc[i]['golden_status']
        cards.append((id, src, type, card_type, status))
    context = {}
    context['title'] = 'golden'
    context['cards'] = cards
    return render(request, 'label.html', context)

def crop_images(request):
    print("cropping images")
    file = 'data_checker/utils/card_links.csv'
    data = pd.read_csv(file)
    if not 'is_cropped' in data:
        data['is_cropped'] = False
    ret_dict = {}
    for index, row in data.iterrows():

        name = misc.get_name(index, row)
        file_name, file_name_golden = misc.get_file_names(index, name, path='data_checker/static/')
        # print(row)
        if row['skip']:
            continue
        if not row['have_golden']:
            continue
        if row['is_cropped']:
            continue

        if row['type'] == 'Minion':
            normal_points = get_points_from_file('data_checker/utils/points/normal_minion_crop_points.txt')
            golden_points = get_points_from_file('data_checker/utils/points/golden_minion_crop_points.txt')
        elif row['type'] == 'Hero Power':
            normal_points = get_points_from_file('data_checker/utils/points/normal_hero_power_crop_points.txt')
            golden_points = get_points_from_file('data_checker/utils/points/golden_hero_power_crop_points.txt')
        elif row['type'] == 'Hero':
            normal_points = get_points_from_file('data_checker/utils/points/normal_hero_crop_points.txt')
            golden_points = get_points_from_file('data_checker/utils/points/golden_hero_crop_points.txt')
        elif row['type'] == 'Ability':
            normal_points = get_points_from_file('data_checker/utils/points/normal_ability_crop_points.txt')
            golden_points = get_points_from_file('data_checker/utils/points/golden_ability_crop_points.txt')
        elif row['type'] == 'Playable Hero':
            normal_points = get_points_from_file('data_checker/utils/points/normal_playable_hero_crop_points.txt')
            golden_points = get_points_from_file('data_checker/utils/points/golden_playable_hero_crop_points.txt')
        elif row['type'] == 'Weapon':
            normal_points = get_points_from_file('data_checker/utils/points/normal_weapon_crop_points.txt')
            golden_points = get_points_from_file('data_checker/utils/points/golden_weapon_crop_points.txt')
        else:
            continue

        card_dir = 'data_checker/utils/cropped_images/{}_{}/'.format(index,name)
        os.makedirs(card_dir, exist_ok=True)
        os.makedirs(card_dir + 'normal', exist_ok=True)
        os.makedirs(card_dir + 'golden', exist_ok=True)
        os.makedirs(card_dir + 'temp', exist_ok=True)

        save_file_normal = card_dir + 'normal/{}_{}_normal_cropped.png'.format(index,name)
        save_file_golden = card_dir + 'golden/{}_{}_golden_cropped'.format(index,name)

        crop_image(file_name, normal_points, save_file=save_file_normal, padding=(200,200))
        processGif(file_name_golden, card_dir + 'temp/{}_{}_temp'.format(index, name))

        i = 0
        listdir = os.listdir(card_dir + 'temp')
        listdir = sorted(listdir, key=lambda x: get_file_number(x))
        # print(listdir)
        for f in listdir:
            # print(f)
            crop_image(card_dir + 'temp/' + f, golden_points , save_file=save_file_golden+'_{}.png'.format(i), padding=(200,200))
            i += 1

        shutil.rmtree(card_dir + 'temp')

        data.loc[index, 'is_cropped'] = True
        # if index >= 1:
        #     break
        ret_dict[name] = 'cropped'
        print("cropped {} {}".format(index, name))
        data.to_csv(file, index=False)
        # break

    return JsonResponse(ret_dict)


def clear_cropped(request):
    file = 'data_checker/utils/card_links.csv'
    data = pd.read_csv(file)
    data['is_cropped'] = False
    data.to_csv(file, index=False)
    return JsonResponse({'is cropped': 'nah'})


def convert_to_dataset(request):
    d = hs_dataset()
    save_file = "cropped_dataset_1.tfrecord"
    save(d, save_file)
    JsonResponse({"save_file":save_file})

def get_data_sizes(request):
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))

    # curr_image_dir = image_dirs[0]
    # normal_cards = []
    # golden_cards = []
    sizes = defaultdict(list)
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
        if golden_image_tensor.shape != (100, 200, 200, 3):
            continue
        sizes[str(normal_image_tensor.shape)].append(normal)
        sizes[str(golden_image_tensor.shape)].append(golden[0])

        # normal_cards.append(normal_image_tensor)
        # golden_cards.append(golden_image_tensor)
        # print(normal_image_tensor.shape)
        # print(golden_image_tensor.shape)
        card_number += 1
    return JsonResponse(sizes)






def delete_unneeded_data(request):
    file = 'data_checker/utils/card_links.csv'
    data = pd.read_csv(file)
    ret_dict = {}
    for index, row in data.iterrows():
        name = misc.get_name(index, row)
        file_name, file_name_golden = misc.get_file_names(index, name, path='data_checker/static/')
        # print(row)
        if row['skip']:
            ret_dict[name] = ""
            if os.path.exists(file_name):
                os.remove(file_name)
                ret_dict[name] += "normal removed"
            else:
                ret_dict[name] += "normal already removed"
            if os.path.exists(file_name_golden):
                os.remove(file_name_golden)
                ret_dict[name] += "\ngolden removed"
            else:
                ret_dict[name] += "\ngolden already removed"
    return JsonResponse(ret_dict)

def delete_cropped_frames(request):
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    removed = {}
    for curr_image_dir in image_dirs:
        shutil.rmtree(str(curr_image_dir) + '/golden/')
        removed[str(curr_image_dir)] = 'removed'
        # golden = [str(g) for g in list(pathlib.Path(curr_image_dir).glob('./golden/*.png'))]
    return JsonResponse(removed)

def delete_numpy_files(request):
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        normal = str(curr_image_dir) + '/' + name + '_normal.npy'
        golden = str(curr_image_dir) + '/' + name + '_golden.npy'
        os.remove(normal)
        os.remove(golden)

def convert_to_gif(request):
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    ret_dict = {}
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        print('converting {} to gif'.format(name))
        golden = [str(g) for g in list(pathlib.Path(curr_image_dir).glob('./golden/*.png'))]
        golden = sorted(golden, key=get_img_number)
        # print(golden)
        frames = [Image.open(f) for f in golden]
        # print(str(curr_image_dir))

        # print('name:', name)
        # os.makedirs(str(curr_image_dir) + '/golden_gif/', exist_ok=True)
        save_file = str(curr_image_dir) + '/' + name + '_golden_cropped.gif'
        # print(save_file)
        frames[0].save(save_file, format='GIF', append_images=frames[1:], save_all=True, duration=len(golden)//2, loop=0)
        ret_dict[name] = 'saved'
        # break
    return JsonResponse(ret_dict)

def move_files(request):
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    for curr_image_dir in image_dirs:
        name = str(curr_image_dir).split('\\')[-1]
        print('moveing {}'.format(name))
        golden_file_name =  str(curr_image_dir) + '/golden_gif/' + name + '_golden_cropped.gif'
        if os.path.exists(golden_file_name):
            shutil.move(golden_file_name, str(curr_image_dir) + '/' + name + '_golden_cropped.gif')
            os.rmdir(str(curr_image_dir) + '/golden_gif/')
        normal_file_name =  str(curr_image_dir) + '/normal/' + name + '_normal_cropped.png'
        if os.path.exists(normal_file_name):
            shutil.move(normal_file_name, str(curr_image_dir) + '/' + name + '_normal_cropped.png')
            os.rmdir(str(curr_image_dir) + '/normal/')
        return JsonResponse({'moved': 'yes'})


def add_classes_to_card_dirs(request):
    data_dir = 'data_checker/utils/cropped_images/'
    image_dirs = list(pathlib.Path(data_dir).glob('*/'))
    print(len(image_dirs))
    file = 'data_checker/utils/card_links.csv'
    data = pd.read_csv(file)
    for dir in image_dirs:
        # print(dir)
        id = re.findall(r'\d+', str(dir).split('\\')[-1])[0]
        type = data.loc[int(id), 'type']
        # print(str(dir) + '_{}'.format(type))
        # os.rename(str(dir), str(dir) + '_{}'.format(type))
    return JsonResponse({'renamed': 'maybe'})


def save_to_file(request):
    # save_file = 'hs_dataset.npz'
    # dataset = hs_dataset()
    save_data_as_npy()
    return JsonResponse({'saved to file': 'saved to lots of files'})
