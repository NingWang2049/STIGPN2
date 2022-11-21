import json
import joblib

# # something_else
# with open('/ssd/datasets/Something-Else/bounding_box_annotations.json' , 'r') as f:
#     anno = json.load(f)

# object_list = []
# for video_id, video_anno in anno.items():
#     for frame_anno in video_anno:
#         object_list += frame_anno['gt_placeholders']
# object_list = list(set(object_list))

# keys = ['book','bowl','box','cloth','cup','medcinebox','microwave','milk','plate','remote','whisk','bottle','banana','cutting board','knife','sponge','hammer','wood','screwdriver']
# for key in keys:
#     for obj in object_list:
#         if obj.find(key) > -1:
#             print(key, '->', obj)

# CAD-120
is_val = False
if not is_val:
    load_dir = '/home/wn/datasets/CAD120/cad_train_data_with_appearence_features.p'
else:
    load_dir = '/home/wn/datasets/CAD120/cad_test_data_with_appearence_features.p'
with open(load_dir,'rb') as f:
    data = joblib.load(f)
sub_activity_list = data['sub_activity_list']
affordence_list = data['affordence_list']
classes_list = data['classes_list']
if not is_val:
    load_data = data['train_data']
else:
    load_data = data['test_data']

affordence_statistics = {}
for video in load_data:
    for key, value in video['label'].items():
        if key != 'person' and value != 'stationary':
            key = key.split('_')[0]
            if key not in affordence_statistics.keys():
                affordence_statistics[key] = set()
            affordence_statistics[key].add(value)

affordence_statistics

for x in classes_list:
    print(x, affordence_statistics[x])