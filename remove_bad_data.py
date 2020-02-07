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



file = 'card_links.csv'
data = pd.read_csv(file)
# print(data.head())

data['skip'] = False

print(data.head())

data.at[3149,'skip'] = True

# total = len(data.index)
# for index, row in data.iterrows():
#     # print(row)
#     if not row['have_golden']:
#         continue
#     if row['is_cropped']:
#         continue
#
#     percent_complete = index/total * 100
#     bar_len = int(percent_complete/5)
#     loading_bar = '█'*bar_len + '-'*(20-bar_len)
#     print('\r {}/{} {} {:.4f}% complete, current: {} '.format(index, total, loading_bar, percent_complete, row['name']), end='\r')
#     # print(index,":",row['name'], row['img_url'], row['golden_img_url'])


data.to_csv(file, index=False)
