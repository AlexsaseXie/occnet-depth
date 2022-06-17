import os
import sys
import time
import datetime
import trimesh
import numpy as np
from tqdm import tqdm
import shutil
import trimesh


OUTPUT_ROOT = '/home3/xieyunwei/ShapeNet.SAL/'
NEW_OUTPUT_ROOT = '/home3/xieyunwei/ShapeNet.SAL.clean/'
BUILD_ROOT = '/home3/xieyunwei/ShapeNet.build.direct_remove/'

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

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a)

def main():
    print('Counting:')
    for model_class in CLASSES:
        missing_count = 0

        class_root = os.path.join(OUTPUT_ROOT, model_class)

        lst_file = os.path.join(class_root, 'train_instance.lst')
        with open(lst_file, 'r') as f:
            model_infos = f.read().split('\n')
            modelnames = [ tmp.split(' ')[1] for tmp in model_infos ]
        
        build_class_root = os.path.join(BUILD_ROOT, model_class, '4_pointcloud_direct_fps_N30000')
        new_build_class_root = os.path.join(BUILD_ROOT, model_class, '4_pointcloud_direct_fps_N30000_copy')

        if not os.path.exists(new_build_class_root):
            os.mkdir(new_build_class_root)

        for model_id in modelnames:
            a = os.path.join(build_class_root, '%s.npz' % model_id)
            b = os.path.join(new_build_class_root, '%s.npz' % model_id)

            shutil.copy(a, b)

            print('%s/%s copy' % (model_class, model_id))

def copy():
    print('Copy')
    mkdir_p(NEW_OUTPUT_ROOT)
    metadata_file = os.path.join(OUTPUT_ROOT, 'metadata.yaml')
    new_metadata_file = os.path.join(NEW_OUTPUT_ROOT, 'metadata.yaml')
    shutil.copy(metadata_file, new_metadata_file)
    for model_class in CLASSES:
        missing_count = 0

        class_root = os.path.join(OUTPUT_ROOT, model_class)
        new_class_root = os.path.join(NEW_OUTPUT_ROOT, model_class)
        mkdir_p(new_class_root)

        lst_file = os.path.join(class_root, 'train_instance.lst')
        new_lst_file = os.path.join(new_class_root, 'train_instance.lst')
        shutil.copy(lst_file, new_lst_file)
        with open(lst_file, 'r') as f:
            model_infos = f.read().split('\n')
            modelnames = [ tmp.split(' ')[1] for tmp in model_infos ]
        
        for model_id in modelnames:
            a = os.path.join(class_root, model_id)
            b = os.path.join(new_class_root, model_id)

            shutil.copytree(a, b)

            print('%s/%s copy' % (model_class, model_id))
      
if __name__ == '__main__':
    #main()
    copy()

