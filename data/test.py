import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm
import json
import re
def sort_key(s):
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)

def strsort(alist):
    alist.sort(key=sort_key)
    return alist

is_val = False

if is_val:
    save_flag = 'val'
    video_info_path = '/raid/wn/Something-else/val_video_info.json'
else:
    save_flag = 'train'
    video_info_path = '/raid/wn/Something-else/train_video_info.json'
with open(video_info_path,'rb') as f:
    video_info = joblib.load(f)
video_ids = list(video_info.keys())
video_ids = strsort(video_ids)
segment = [int(video_ids[15000].split('.')[0]),int(video_ids[30000].split('.')[0]),int(video_ids[45000].split('.')[0])]
data_store1 = pd.HDFStore('/raid/wn/Something-else/'+save_flag+'_data_1.h5')
data_store2 = pd.HDFStore('/raid/wn/Something-else/'+save_flag+'_data_2.h5')
data_store3 = pd.HDFStore('/raid/wn/Something-else/'+save_flag+'_data_3.h5')
data_store4 = pd.HDFStore('/raid/wn/Something-else/'+save_flag+'_data_4.h5')

for folder_id in tqdm(video_ids):
    folder_id = folder_id.split('.')[0]
    if is_val:
        video_path = '/ssd2/wn/something-else/val/'+folder_id+'.pkl'
    else:
        video_path = '/ssd2/wn/something-else/train/'+folder_id+'.pkl'
    with open(video_path,'rb') as f:
        data = joblib.load(f)
    video_node_features = data['video_node_features']

for folder_id in tqdm(video_ids):
    folder_id = folder_id.split('.')[0]
    if int(folder_id) < segment[0]:
        video_node_features = np.array(data_store1['video_'+folder_id]).reshape(4,-1,2048)
    elif int(folder_id) < segment[1]:
        video_node_features = np.array(data_store2['video_'+folder_id]).reshape(4,-1,2048)
    elif int(folder_id) < segment[2]:
        video_node_features = np.array(data_store3['video_'+folder_id]).reshape(4,-1,2048)
    else:
        video_node_features = np.array(data_store4['video_'+folder_id]).reshape(4,-1,2048)