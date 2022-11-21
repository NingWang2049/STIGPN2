import os
import joblib
import json
import torch
from glob import glob
import numpy as np
import cv2
import math
from skimage.transform import resize
import argparse
from tqdm import tqdm
import gc

parser = argparse.ArgumentParser(description='PyTorch Smth-Else')
# Path related arguments
parser.add_argument('--fetch_flag',default='train')
parser.add_argument('--start_idx',default='0')
args = parser.parse_args()

root = '/home/wn/datasets/Something-Something/'
root_frames = '/ssd2/sth2sth/frame/'
train_json_path = os.path.join(root,'annotations/compositional/train.json')
validation_json_path = os.path.join(root,'annotations/compositional/validation.json')
labels_json_path = os.path.join(root,'annotations/compositional/labels.json')
bounding_box_annotations_path = '/ssd2/wn/something-else/'+args.fetch_flag+'.json'#/ssd2/wn/something-else/bounding_box_annotations.json
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

start_idx = int(args.start_idx)
class VideoFolder(torch.utils.data.Dataset):
    def __init__(self,fetch_flag,bounding_box_annotations,start_idx=0):
        self.fetch_flag = fetch_flag
        self.num_nodes = 4
        self.bounding_box_annotations = bounding_box_annotations
        if self.fetch_flag == 'train':
            with open(train_json_path,'rb') as f:
                self.data_jsons = json.load(f)
        elif self.fetch_flag == 'val':
            with open(validation_json_path,'rb') as f:
                self.data_jsons = json.load(f)
        self.start_idx = start_idx
    
    def __len__(self):
        return len(self.data_jsons)

    def process(self,num_frames,frame_list,video_anno,object_set):
        batch_segments = np.zeros([1, self.num_nodes, num_frames, 224, 224, 3])*1.0
        for frame_idx,frame_id in enumerate(frame_list):
            frame_anno = video_anno[frame_id]
            frame_name = frame_anno['name']
            frame_labels = frame_anno['labels']
            img_path = os.path.join(root_frames,frame_name)
            im_orig  = cv2.imread(img_path)
            if im_orig is None:
                im_orig = prev_im
                print('Image not found, using previous frame')
            else:
                prev_im = im_orig
            im_orig  = im_orig.astype(np.float32)/256.0
            img_wh = [im_orig.shape[1],im_orig.shape[0]]
            for label in frame_labels:
                standard_category = label['standard_category']
                object_index = object_set.index(standard_category)
                x1, x2, y1, y2 = label['box2d']['x1'], label['box2d']['x2'], label['box2d']['y1'], label['box2d']['y2']
                x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.floor(x2), math.floor(y2)
                img_cropped = im_orig[y1:y2,x1:x2]
                if img_cropped.shape[0] == 0 or img_cropped.shape[1] == 0:
                    continue
                if object_index < self.num_nodes:
                    batch_segments[0,object_index,frame_idx] = resize(img_cropped, (224, 224, 3))
        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        batch_segments[:, :self.num_nodes] = (batch_segments[:, :self.num_nodes] - img_mean)/img_std
        batch_segments = torch.tensor(batch_segments).float()
        return batch_segments[0],img_wh
    
    def __getitem__(self, index):
        if index < self.start_idx:
            return torch.zeros((1)),torch.zeros((1)),torch.zeros((1))
        data_json = self.data_jsons[index]
        video_id = data_json['id']
        video_anno = self.bounding_box_annotations[video_id]
        num_frames = len(video_anno)
        frame_list = [x for x in range(num_frames)]
        # union the objects of two frames
        object_set = set()
        for frame_id in frame_list:
            try:
                frame_data = video_anno[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                object_set.add(standard_category)
        object_set = sorted(list(object_set))
        if 'hand' in object_set:
            object_set.remove('hand')
            object_set.insert(0,'hand')
        node_im_segments,img_wh = self.process(num_frames,frame_list,video_anno,object_set)
        return video_id,node_im_segments,img_wh

res50 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
res50 = torch.nn.Sequential(*(list(res50.children())[:-1]))
res50.cuda()
res50.eval()


with open(bounding_box_annotations_path,'rb') as f:
    bounding_box_annotations = json.load(f)

dataset = VideoFolder(fetch_flag=args.fetch_flag,bounding_box_annotations=bounding_box_annotations,start_idx=start_idx)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=4)

count = 0
for video_id,node_im_segments,img_wh in tqdm(data_loader):
    if count < start_idx:
        count += 1
        continue
    video_id = video_id[0]
    node_im_segments = node_im_segments[0].cuda()
    img_wh = img_wh[0].numpy()[0],img_wh[1].numpy()[0]
    node_im_segments = node_im_segments.permute(0, 1, 4, 2, 3)
    nodes,frames = node_im_segments.shape[0],node_im_segments.shape[1]
    res50_feats = np.zeros((nodes, frames, 2048, 1, 1))
    for o in range(nodes):
        with torch.no_grad():
            res50_feats[o, :] = res50(node_im_segments[o, :]).cpu().numpy()
    video_node_features = res50_feats.squeeze(3).squeeze(3)
    save_path = '/ssd2/wn/something-else/'+args.fetch_flag+'/'+video_id+'.pkl'
    data = {}
    data['video_node_features'] = video_node_features
    data['img_wh'] = img_wh
    with open(save_path,'wb') as f:
        joblib.dump(data,f)