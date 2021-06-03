# import mechanicalsoup
import pandas as pd
import numpy as np
import cv2
from PIL import Image, GifImagePlugin, ImageDraw, ImageSequence
# import urllib
import requests
import shutil
import sys
import re

from os import path, stat
import os

def get_points_from_file(file_name):
    arr = []
    with open(file_name) as f:
        for line in f:
            # arr.append((int(i) for i in line.split(',')))
            x,y = line.split(',')
            xy = (int(x), int(y))
            arr.append(xy)
            # arr.append(int(y))
    return arr

def create_mask(width, height, points):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # draw.ellipse((140,50,260,170), fill=255)
    draw.polygon(points, fill=255)
    # mask.show()
    return mask


def crop_image_with_mask(img, mask, box):
    new_img = Image.new('RGBA', img.size, 0)
    masked = Image.composite(img, new_img, mask)
    return masked.crop(box)

def bounding_box(points):
    x_coords, y_coords = zip(*points)
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def crop_image(file, points, save_file=None, padding=None):
    img = Image.open(file)
    w, h = img.size
    mask = create_mask(w, h, points)
    box = bounding_box(points)
    cropped = crop_image_with_mask(img, mask, box)
    if padding:
        fin = Image.new('RGBA',padding)
        x1, y1 = cropped.size
        x = padding[0]//2 - x1//2
        y = padding[1]//2 - y1//2
        coords = [x,y,x+x1,y+y1]
        fin.paste(cropped, coords)
    else:
        fin = cropped
    # fin.show()
    if save_file:
        fin.save(save_file)
    return fin

def processGif(infile, save_file):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)
    i = 0
    for frame in ImageSequence.Iterator(im):
        # if (i - 12) >= -1 and (i - 12) <= 1:
        #     frame.show()
        frame.save(save_file+ "_"+str(i)+'.png')
        i += 1
    # i = 0
    # mypalette = im.getpalette()
    #
    # try:
    #     while 1:
    #         im.putpalette(mypalette)
    #         new_im = Image.new("RGBA", im.size)
    #         new_im.paste(im)
    #         new_im.save(save_file+ "_"+str(i)+'.png')
    #
    #         i += 1
    #         im.seek(im.tell() + 1)
    #
    # except EOFError:
    #     pass # end of sequence

def get_file_number(f):
    return int(re.findall(r'\d+\.png',f)[0][:-4])

def get_crop_points(type, style):
    if type == 'Minion':
        if style == "normal":
            return np.array(get_points_from_file('data_checker/utils/points/normal_minion_crop_points.txt'))
        else:
            return np.array(get_points_from_file('data_checker/utils/points/golden_minion_crop_points.txt'))
    elif type == 'Hero Power':
        if style == "normal":
            return np.array(get_points_from_file('data_checker/utils/points/normal_hero_power_crop_points.txt'))
        else:
            return np.array(get_points_from_file('data_checker/utils/points/golden_hero_power_crop_points.txt'))
    elif type == 'Hero':
        if style == "normal":
            return np.array(get_points_from_file('data_checker/utils/points/normal_hero_crop_points.txt'))
        else:
            return np.array(get_points_from_file('data_checker/utils/points/golden_hero_crop_points.txt'))
    elif type == 'Playable Hero':
        if style == "normal":
            return np.array(get_points_from_file('data_checker/utils/points/normal_playable_hero_crop_points.txt'))
        else:
            return np.array(get_points_from_file('data_checker/utils/points/golden_playable_hero_crop_points.txt'))
    elif type == 'Weapon':
        if style == "normal":
            return np.array(get_points_from_file('data_checker/utils/points/normal_weapon_crop_points.txt'))
        else:
            return np.array(get_points_from_file('data_checker/utils/points/golden_weapon_crop_points.txt'))
    elif type == 'Ability':
        if style == "normal":
            return np.array(get_points_from_file('data_checker/utils/points/normal_ability_crop_points.txt'))
        else:
            return np.array(get_points_from_file('data_checker/utils/points/golden_ability_crop_points.txt'))
    else:
        return Exception("not valid points")










def crop():
    file = 'card_links.csv'
    data = pd.read_csv(file)
    # print(data.head())

    total = len(data.index)


    for index, row in data.iterrows():
        # print(row)
        if row['skip']:
            continue
        if not row['have_golden']:
            continue
        if row['is_cropped']:
            continue

        percent_complete = index/total * 100
        bar_len = int(percent_complete/5)
        loading_bar = '█'*bar_len + '-'*(20-bar_len)
        print('\r {}/{} {} {:.4f}% complete, current: {} '.format(index, total, loading_bar, percent_complete, row['name']), end='\r')
        # print(index,":",row['name'], row['img_url'], row['golden_img_url'])

        name = "unknown_{}".format(index)
        if not pd.isna(row['name']):
            name = row['name']

        bad_chars = '*.\"\\\/[]:;|,!?'
        for c in bad_chars:
            name = name.replace(c,'')

        file_name_golden = 'golden_cards/{}_{}_golden.gif'.format(index, name)
        file_name = 'normal_cards/{}_{}_normal.png'.format(index, name)

        normal_points = []
        golden_points = []

        if row['type'] == 'Minion':
            normal_points = np.array(get_points_from_file('points/normal_minion_crop_points.txt'))
            golden_points = np.array(get_points_from_file('points/golden_minion_crop_points.txt'))
        elif row['type'] == 'Hero Power':
            normal_points = np.array(get_points_from_file('points/normal_hero_power_crop_points.txt'))
            golden_points = np.array(get_points_from_file('points/golden_hero_power_crop_points.txt'))
        elif row['type'] == 'Hero':
            normal_points = np.array(get_points_from_file('points/normal_hero_crop_points.txt'))
            golden_points = np.array(get_points_from_file('points/golden_hero_crop_points.txt'))
        elif row['type'] == 'Playable Hero':
            normal_points = np.array(get_points_from_file('points/normal_playable_hero_crop_points.txt'))
            golden_points = np.array(get_points_from_file('points/golden_playable_hero_crop_points.txt'))
        elif row['type'] == 'Weapon':
            normal_points = np.array(get_points_from_file('points/normal_weapon_crop_points.txt'))
            golden_points = np.array(get_points_from_file('points/golden_weapon_crop_points.txt'))
        elif row['type'] == 'Ability':
            normal_points = np.array(get_points_from_file('points/normal_ability_crop_points.txt'))
            golden_points = np.array(get_points_from_file('points/golden_ability_crop_points.txt'))
        else:
            continue

        card_dir = 'cropped_images/{}_{}/'.format(index,name)
        os.makedirs(card_dir, exist_ok=True)
        os.makedirs(card_dir + 'normal', exist_ok=True)
        os.makedirs(card_dir + 'golden', exist_ok=True)
        os.makedirs(card_dir + 'temp', exist_ok=True)

        save_file_normal = card_dir + 'normal/{}_{}_normal_cropped.png'.format(index,name)
        save_file_golden = card_dir + 'golden/{}_{}_golden_cropped'.format(index,name)

        crop_image(normal_points, file_name, save_file_normal)
        processGif(file_name_golden, card_dir + 'temp/{}_{}_temp'.format(index, name))

        i = 0
        for f in os.listdir(card_dir + 'temp'):
            # print(f)
            crop_image(golden_points, card_dir + 'temp/' + f, save_file_golden + '_{}.png'.format(i))
            i += 1

        shutil.rmtree(card_dir + 'temp')

        data.at[index, 'is_cropped'] = True
        data.to_csv(file, index=False)

if __name__ =='__main__':
    crop()
