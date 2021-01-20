import os
import sys
import random
import subprocess
import multiprocessing
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import re

sys.path.append('./')
from im2mesh.utils.depth_to_pointcloud import DepthToPCNp
from im2mesh.utils.visualize import visualize_pointcloud

# Arguments
parser = argparse.ArgumentParser(
    description='Generate point cloud from depth'
)
parser.add_argument('--mask_dir', type=str, default='./data/pix3d/generation/')
parser.add_argument('--depth_dir', type=str, default='./data/pix3d/uresnet.depth_pred/', help='depth & output dir')
parser.add_argument('--out_folder_name', type=str, default='depth_pointcloud', help='output folder name')
parser.add_argument('--pix3d_root', type=str, default='.', help='pix3d_root which is not necessary')
parser.add_argument('--nproc', type=int, default=10, help='parallel process num')
parser.add_argument('--n', type=int, default=2048, help='subsample point num N')
parser.add_argument('--task_split_root', type=str, default='./scripts/pix3d_preprocess/task_split')
parser.add_argument('--test_root', type=str, default='./data/back_projection_pix3d_test/')
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()

MASK_ROOT = args.mask_dir
DEPTH_ROOT = args.depth_dir
depth_pred = re.split('.|/', DEPTH_ROOT)
OUTPUT_DIR_NAME = args.out_folder_name
N = args.n

TEST_ROOT = args.test_root

CLASSES = [
    'sofa',
    'chair',
    'table'
]

TASK_SPLIT_ROOT = args.task_split_root
NPROC = args.nproc

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

def back_projection(task_file, task_i):
    start_time = time.time()
    print('Render start:', task_i)

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

    if task_i == 0:
        all_model_info = tqdm(all_model_info)

    for model_info in all_model_info:
        model_class = model_info[0]
        model_id = model_info[1]

        depth_folder = os.path.join(DEPTH_ROOT, model_class, model_id, depth_foldername)
        mask_folder = os.path.join(MASK_ROOT, model_class, model_id)

        output_folder = os.path.join(DEPTH_ROOT, model_class, model_id, OUTPUT_DIR_NAME)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        depth_range_file = os.path.join(depth_folder, 'depth_range.txt')
        
        with open(depth_range_file, 'r') as f:
            depth_range = f.readline().split(' ')
            depth_min = float(depth_range[0])
            depth_max = float(depth_range[1])
            depth_unit = float(depth_range[2])                

            depth_file = os.path.join(depth_folder, '00_depth.png')
            mask_file = os.path.join(mask_folder, '%s_final_mask.png' % model_id)
            depth_img = Image.open(depth_file).convert('L')
            mask_img = Image.open(mask_file)
            pts = worker.work(depth_img, mask_img, depth_min, depth_max, n=N, unit=depth_unit)

            output_file = os.path.join(output_folder, '00_pointcloud.npz')
            np.savez(output_file, pointcloud=pts)    
    
    end_time = time.time() 
    print('Render end:', task_i, ',cost:', end_time - start_time)

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
    model_class = '02691156'
    model_id = '10155655850468db78d106ce0a280f87'

    if depth_pred:
        depth_foldername = 'depth_pred'
    else:
        depth_foldername = 'depth'

    depth_folder = os.path.join(DEPTH_ROOT, model_class, model_id, depth_foldername)
    mask_folder = os.path.join(MASK_ROOT, model_class, model_id)

    output_folder = os.path.join(TEST_ROOT)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    depth_range_file = os.path.join(depth_folder, 'depth_range.txt')
    worker = DepthToPCNp()

    with open(depth_range_file, 'r') as f:
        depth_range = f.readline().split(' ')
        depth_min = float(depth_range[0])
        depth_max = float(depth_range[1])
        depth_unit = float(depth_range[2]) 

        depth_file = os.path.join(depth_folder, '00_depth.png')
        mask_file = os.path.join(mask_folder, '%s_final_mask.png' % model_id)
        depth_img = Image.open(depth_file).convert('L')
        depth_img.save(os.path.join(output_folder, '00_depth.png'))
        mask_img = Image.open(mask_file)
        pts = worker.work(depth_img, mask_img, depth_min, depth_max, n=N, unit=depth_unit)

        output_file = os.path.join(output_folder, '00_pointcloud.npz')
        np.savez(output_file, pointcloud=pts)
        
        output_file = os.path.join(output_folder, '00_pc.png')
        visualize_pointcloud(pts, out_file=output_file, show=True, elev=15, azim=180)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.test:
        test()
    else:
        main()
