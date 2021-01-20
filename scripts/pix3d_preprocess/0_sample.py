import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import utils

parser = argparse.ArgumentParser('Test keypoints')
parser.add_argument('--pix3d_root', type=str, default='.')
parser.add_argument('--sampled_list_root', type=str, default='./sampled_list/')
parser.add_argument('--focal', action='store_true') # this is not a good criterion
args = parser.parse_args()

TEST_CLASSES = [
    'sofa',
    'chair',
    'table'
]

test_class_infos = {
    'sofa': [],
    'chair': [],
    'table': []
}

PIX3D_ROOT = args.pix3d_root
SAMPLED_LIST_ROOT = args.sampled_list_root

with open(os.path.join(PIX3D_ROOT, 'pix3d.json')) as f:
    all_infos = json.load(f)

remove_count = {
    'sofa': [0,0,0,0,0,0,0],
    'chair': [0,0,0,0,0,0,0],
    'table': [0,0,0,0,0,0,0]
}

# selecting images
for info in all_infos:
    if info['category'] in TEST_CLASSES:
        remove_count[info['category']][-1] += 1

        # remove truncated or occluded images
        if info['truncated'] or info['occluded'] or info['slightly_occluded']:
            remove_count[info['category']][0] += 1
            continue

        # tranlation along x y should be small
        trans_mat = info['trans_mat']
        if abs(trans_mat[0]) >= 0.02 * abs(trans_mat[2]) or \
            abs(trans_mat[1]) >= 0.02 * abs(trans_mat[2]):
            remove_count[info['category']][1] += 1
            continue

        # inplane_rotation should be small
        if abs(info['inplane_rotation']) * 180. / np.pi >= 5:
            remove_count[info['category']][2] += 1
            continue

        # make sure enough keypoints annotations available
        # get keypoints
        two_d_keypoints = np.array(info['2d_keypoints'])
        
        # combine all annotations
        num_keypoints = two_d_keypoints.shape[1]
        remove_ = (two_d_keypoints[:,:,:] < 0).any(axis=(0,2))
        #print('remove_:', remove_, remove_.shape)
        two_d_keypoints = np.mean(two_d_keypoints, axis=0)
        two_d_keypoints[remove_ , :] = -1.

        two_d_keypoints_count = (two_d_keypoints > 0.).sum() / 2.
        if two_d_keypoints_count <= 5:
            remove_count[info['category']][3] += 1 
            continue

        # image size
        if info['img_size'][0] <= 300 or info['img_size'][1] <= 300:
            remove_count[info['category']][4] += 1 
            continue

        # focal length
        if args.focal and (info['focal_length'] < 30 or info['focal_length'] > 40):
            remove_count[info['category']][5] += 1
            continue

        test_class_infos[info['category']].append(info)

for test_cls in TEST_CLASSES:
    all_count = len(test_class_infos[test_cls]) 
    print('-----CLASS', test_cls, ' remains:', len(test_class_infos[test_cls]))

    print('Truncated or occluded removed:', remove_count[test_cls][0])
    print('Tranlation removed:', remove_count[test_cls][1])
    print('Inplane rotation removed:', remove_count[test_cls][2])
    print('2D keypoint removed:', remove_count[test_cls][3])
    print('Img size remove:', remove_count[test_cls][4])
    if args.focal:
        print('Focal length remove:', remove_count[test_cls][5])

    print('All:', remove_count[test_cls][-1])

if not os.path.exists(SAMPLED_LIST_ROOT):
    os.mkdir(SAMPLED_LIST_ROOT)

# dump list
for test_cls in TEST_CLASSES:
    cls_infos = test_class_infos[test_cls]
    cls_sampling_root = os.path.join(SAMPLED_LIST_ROOT, test_cls)

    if not os.path.exists(cls_sampling_root):
        os.mkdir(cls_sampling_root)

    with open(os.path.join(cls_sampling_root, 'all_info.json'), 'w') as f:
        json.dump(cls_infos, f)
    