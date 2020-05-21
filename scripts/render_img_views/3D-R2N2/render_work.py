import os
import sys

SHAPENET_ROOT = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/'
DIR_RENDERING_PATH = '/home2/xieyunwei/occupancy_networks/data/render'

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

NPROC = 15

def split_task():
    if not os.path.exists(DIR_RENDERING_PATH):
        os.mkdir(DIR_RENDERING_PATH)

    if not os.path.exists(TASK_SPLIT_ROOT):
        os.mkdir(TASK_SPLIT_ROOT)
        
    all_model_count = 0
    all_model_classes = []
    all_model_ids = []

    for model_class in CLASSES:
        class_root = os.path.join(SHAPENET_ROOT, model_class)
        current_class_ids = os.listdir(class_root)
        
        all_model_count += len(current_class_ids)
        all_model_ids += current_class_ids
        all_model_classes += [ model_class for model_id in current_class_ids]

    split_number = (int) (all_model_count / NPROC)
    for i in range(NPROC):
        if i != NPROC - 1:
            i_task_model_classes = all_model_classes[i * split_number : (i+1) * split_number]
            i_task_model_ids = all_model_ids[i * split_number : (i+1) * split_number]
        else:
            i_task_model_classes = all_model_classes[i * split_number :]
            i_task_model_ids = all_model_ids[i * split_number :]
        with open(os.path.join(TASK_SPLIT_ROOT,'%d.txt' % (i)),'w') as f:
            for i, model_class in enumerate(i_task_model_classes):
                print('%s %s' % (i_task_model_classes[i], i_task_model_ids[i]), file = f)

    print('All model count:', all_model_count)
    print('Split into:', NPROC, 'tasks')

    for model_class in CLASSES:
        class_root = os.path.join(DIR_RENDERING_PATH,model_class)
        if not os.path.exists(class_root):
            os.mkdir(class_root)
    print('Dirs created')

import subprocess
import multiprocessing
import time

def render_obj(task_file, i):
    start_time = time.time()
    #print('Render start:', i)
    p = subprocess.Popen(['blender','-b','--python','r2n2_render_blender.py','--','--task_file', task_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    standard_out = p.stdout.readlines()
    print(standard_out)

    end_time = time.time() 
    print('Render end:', i, ',cost:', end_time - start_time)

def get_rgb_depth_images(task_file, i):
    start_time = time.time()
    #print('Transfer start:', i)

    p = subprocess.Popen(['python', 'openexr_to_png.py', '--task_file', task_file],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    standard_out = p.stdout.readlines()

    end_time = time.time() 
    print('Transfer end:', i, ',cost:', end_time - start_time)

def main():
    split_task()

    pool = multiprocessing.Pool(processes = NPROC)

    for i in range(NPROC):
        task_file = str(os.path.join(TASK_SPLIT_ROOT, '%d.txt' % i))
        print('render:', task_file)
        pool.apply_async(render_obj, (task_file, i) )
        
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(processes = NPROC)

    for i in range(NPROC):
        task_file = str(os.path.join(TASK_SPLIT_ROOT, '%d.txt' % i))
        print('transfer:', task_file)
        pool.apply_async(get_rgb_depth_images, (task_file, i) )
    
    pool.close()
    pool.join()
    print('finished!') 


if __name__ == '__main__':
    main()   
