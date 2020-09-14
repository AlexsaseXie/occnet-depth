import os
import sys
import random
import subprocess
import multiprocessing
import time
import argparse
import numpy as np
from PIL import Image
from im2mesh.utils.depth_to_pointcloud import DepthToPCNp

MASK_ROOT = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.with_depth.10w10w'
DEPTH_ROOT = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.depth_pred.origin_subdivision'
depth_pred = 'depth_pred' in DEPTH_ROOT.split('.')
OUTPUT_DIR_NAME = 'depth_pointcloud'
N = 2048

TEST_ROOT = './data/back_projection_test/'

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

TASK_SPLIT_ROOT = '/home2/xieyunwei/occupancy_networks/scripts/render_img_views/3D-R2N2/task_split'
NPROC = 10
N_VIEWS = 24

def split_task():
    if not os.path.exists(TASK_SPLIT_ROOT):
        os.mkdir(TASK_SPLIT_ROOT)
        
    all_model_count = 0
    all_model_info = []

    for model_class in CLASSES:
        class_root = os.path.join(DEPTH_ROOT, model_class)
        current_class_ids = os.listdir(class_root)

        #check if model.obj exists
        for model_id in current_class_ids:
            folder = os.path.join(DEPTH_ROOT, model_class, model_id)
            
            if os.path.isdir(folder):
                all_model_count += 1
                all_model_info.append( [model_class, model_id] )

    # save all tasks
    with open(os.path.join(TASK_SPLIT_ROOT,'all.txt'), 'w') as f:
        for info in all_model_info:
            print('%s %s' % (info[0], info[1]), file = f)

    # shuffle
    random.shuffle(all_model_info)
    split_number = (int) (all_model_count / NPROC)
    for i in range(NPROC):
        if i != NPROC - 1:
            i_task_model_info = all_model_info[i * split_number: (i+1) * split_number]
        else:
            i_task_model_info = all_model_info[i * split_number:]
        with open(os.path.join(TASK_SPLIT_ROOT,'%d.txt' % (i)),'w') as f:
            for info in i_task_model_info:
                print('%s %s' % (info[0], info[1]), file = f)

    print('All model count:', all_model_count)
    print('Split into:', NPROC, 'tasks')

    return all_model_info

def back_projection(task_file, i):
    start_time = time.time()
    print('Render start:', i)

    if depth_pred:
        depth_foldername = 'depth_pred'
    else:
        depth_foldername = 'depth'

    all_model_info = []

    with open(task_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '': 
                continue

            tmp = line.rstrip('\n').split(' ')
            all_model_info.append([tmp[0], tmp[1]])

    worker = DepthToPCNp()
    for model_info in all_model_info:
        model_class = model_info[0]
        model_id = model_info[1]

        depth_folder = os.path.join(DEPTH_ROOT, model_class, model_id, depth_foldername)
        mask_folder = os.path.join(MASK_ROOT, model_class, model_id, 'mask')

        output_folder = os.path.join(DEPTH_ROOT, model_class, model_id, OUTPUT_DIR_NAME)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for i in range(N_VIEWS):
            depth_file = os.path.join(depth_folder, '%2d_depth.png' % i)
            mask_file = os.path.join(mask_folder, '%2d_mask.png' % i)
            depth_img = Image.open(depth_file)
            mask_img = Image.open(mask_file)
            pts = worker.work(depth_img, mask_img, n=N, unit=1.)

            output_file = os.path.join(output_folder, '%2d_pointcloud.npz')
            np.savez(output_file, pointcloud=pts)    
    
    end_time = time.time() 
    print('Render end:', i, ',cost:', end_time - start_time)

def main():
    _ = split_task()    

    # back projection
    start_time = time.time()
    process_array = []
    for i in range(NPROC):
        task_file = str(os.path.join(TASK_SPLIT_ROOT, '%d.txt' % i))
        print('Back projection:', task_file)
        p = multiprocessing.Process(target=back_projection, args=(task_file,i))
        p.start()
        process_array.append(p)

    for i in range(NPROC):
        process_array[i].join()
    
    end_time = time.time()
    print('Back projection finished in %f sec' % (end_time - start_time))

    print('finished!')   

def test():
    model_class = '02958343'
    model_id = 'f9c1d7748c15499c6f2bd1c4e9adb41'

    if depth_pred:
        depth_foldername = 'depth_pred'
    else:
        depth_foldername = 'depth'

    depth_folder = os.path.join(DEPTH_ROOT, model_class, model_id, depth_foldername)
    mask_folder = os.path.join(MASK_ROOT, model_class, model_id, 'mask')

    output_folder = os.path.join(TEST_ROOT)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    worker = DepthToPCNp()
    for i in range(N_VIEWS):
        depth_file = os.path.join(depth_folder, '%2d_depth.png' % i)
        mask_file = os.path.join(mask_folder, '%2d_mask.png' % i)
        depth_img = Image.open(depth_file)
        mask_img = Image.open(mask_file)
        pts = worker.work(depth_img, mask_img, n=N, unit=1.)

        output_file = os.path.join(output_folder, '%2d_pointcloud.npz')
        np.savez(output_file, pointcloud=pts)    


parser = argparse.ArgumentParser(description='Back projection')
parser.add_argument('--test', action='store_true', help='test')
if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        test()
    else:
        main()
