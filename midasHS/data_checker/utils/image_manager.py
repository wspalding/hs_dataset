from data_checker.utils.download_cards import file, dir_name_format, file_name_format, path
import pandas as pd
import glob


def get_normal_img_src(row):
    id = int(row['card_id'])
    type = row.iloc[0]['card_type']
    file_name = file_name_format.format(id=id, style='normal', filetype='png')
    dir_name = path + dir_name_format.format(id=id, type=type, datasets='*')
    files = glob.glob(dir_name + file_name)
    if len(files) <= 0:
        url = row.iloc[0]['img_url']
        return url, 'url_img'
    file_path = str(files[0])
    file_path = file_path.replace(path, '')
    return file_path, 'path'



def get_golden_img_src(row):
    id = int(row['card_id'])
    type = row.iloc[0]['card_type']
    file_name = file_name_format.format(id=id, style='golden', filetype='.*')
    dir_name = path + dir_name_format.format(id=id, type=type, datasets='*')
    files = glob.glob(dir_name + file_name)
    if len(files) <= 0:
        url = row.iloc[0]['golden_img_url']
        return url, 'url_video'
    file_path = str(files[0])
    file_path = file_path.replace(path, '')
    return file_path, 'path'

def get_normal_img_path_from_id(id):
    data = pd.read_csv(file)
    row = data.loc[data['card_id'] == int(id)]
    type = row.iloc[0]['card_type']
    file_name = file_name_format.format(id=id, style='normal', filetype='png')
    dir_name = path + dir_name_format.format(id=id, type=type, datasets='*')
    files = glob.glob(dir_name + file_name)
    if len(files) <= 0:
        url = row.iloc[0]['img_url']
        return url, 'url_img'
    file_path = str(files[0])
    file_path = file_path.replace(path, '')
    return file_path, 'path'
