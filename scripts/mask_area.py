import os
import sys
import time
import datetime

from PIL import Image
import numpy as np
from tqdm import tqdm

SHAPENET_ROOT = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/'
R2N2_ROOT = '/home2/xieyunwei/occupancy_networks/external/Choy2016/ShapeNetRendering/'
DATASET_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.with_depth.10w10w/'

CLASSES = [
    '03001627',
    '02958343',
    '04256520',
    '02691156',
    '03636649',
    '04401088',
    '04530566',
    '03691459',
    '02933112',
    '04379243',
    '03211117',
    '02828884',
    '04090263',
]

N_VIEWS = 24

def main():
    print('Counting:')
    for model_class in CLASSES:
        area_list = []

        class_root = os.path.join(DATASET_PATH, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = [ name for name in os.listdir(class_root) 
                              if os.path.isdir(os.path.join(class_root, name)) ]

        for model_id in tqdm(current_class_ids):
            model_root = os.path.join(DATASET_PATH, model_class, model_id)
            view_id = np.random.randint(24,size=1)
            
            mask_path = os.path.join(model_root, 'mask', '%02d_mask.png' % view_id)
            mask = Image.open(mask_path).convert('1')
            mask_np = np.array(mask)
            area_list.append(mask_np.sum())

        print('Class %s: min:%d, max:%d, avg:%f' % ( model_class, min(area_list), max(area_list), sum(area_list) / len(area_list) ) )

if __name__ == '__main__':
    main()

