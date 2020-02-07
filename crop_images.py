# import mechanicalsoup
import pandas as pd
import numpy as np
import cv2
from PIL import Image, GifImagePlugin
# import urllib
import requests
import shutil
import sys

from os import path, stat
import os

def get_points_from_file(file_name):
    arr = []
    with open(file_name) as f:
        for line in f:
            arr.append([int(i) for i in line.split(',')])
    return arr

def crop_image(pts, img_file, save_file):
    img = cv2.imread(img_file)
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()


    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    cv2.imwrite(save_file, dst)
    return

def processGif(infile, save_file):
    try:
        im = Image.open(infile)
    except IOError:
        print("Cant load", infile)
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            new_im.save(save_file+ "_"+str(i)+'.png')

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence

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
