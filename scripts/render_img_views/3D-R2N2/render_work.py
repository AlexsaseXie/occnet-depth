import os
import sys
import random
import subprocess
import multiprocessing
import time

SHAPENET_ROOT = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/'
R2N2_ROOT = '/home2/xieyunwei/occupancy_networks/external/Choy2016/ShapeNetRendering/'
DIR_RENDERING_PATH = '/home2/xieyunwei/occupancy_networks/data/render_2'

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
    if not os.path.exists(DIR_RENDERING_PATH):
        os.mkdir(DIR_RENDERING_PATH)

    if not os.path.exists(TASK_SPLIT_ROOT):
        os.mkdir(TASK_SPLIT_ROOT)
        
    all_model_count = 0
    all_model_info = []

    for model_class in CLASSES:
        #class_root = os.path.join(SHAPENET_ROOT, model_class)
        class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_root)

        #check if model.obj exists
        for model_id in current_class_ids:
            obj_path = os.path.join(SHAPENET_ROOT, model_class, model_id, 'model.obj')
            
            if os.path.exists(obj_path):
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

    for model_class in CLASSES:
        class_root = os.path.join(DIR_RENDERING_PATH,model_class)
        if not os.path.exists(class_root):
            os.mkdir(class_root)
    print('Dirs created')

    return all_model_info

def render_all_obj(task_file, thread):
    p = subprocess.Popen(['blender','-b','--python','r2n2_render_blender.py','-t %d' % thread,'--','--task_file', task_file], stdout=open('/dev/null','w'), stderr=subprocess.STDOUT)
    p.wait()
    print('finished')


def render_obj(task_file, i):
    start_time = time.time()
    print('Render start:', i)
    p = subprocess.Popen(['blender','-b','--python','r2n2_render_blender.py','--','--task_file', task_file], stdout=open('/dev/null','w'), stderr=subprocess.STDOUT)
    p.wait()

    end_time = time.time() 
    print('Render end:', i, ',cost:', end_time - start_time)

def render_single_obj(model_class, model_id):
    p = subprocess.Popen(['blender','-b','--python','r2n2_render_blender.py','--', '--single', '--model_class', model_class, '--model_id', model_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

def get_rgb_depth_images(task_file, i):
    start_time = time.time()
    print('Transfer start:', i)

    p = subprocess.Popen(['python', 'openexr_to_png.py', '--task_file', task_file],stdout=open('/dev/null','w'),stderr=subprocess.STDOUT)
    p.wait()

    end_time = time.time() 
    print('Transfer end:', i, ',cost:', end_time - start_time)

def get_single_rgb_depth_images(model_class, model_id):
    p = subprocess.Popen(['python', 'openexr_to_png.py', '--single', '--model_class', model_class, '--model_id', model_id],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out, err = p.communicate()

def get_mask(task_file, i):
    start_time = time.time()
    print('Save mask start:', i)

    p = subprocess.Popen(['python', 'save_mask.py', '--task_file', task_file],stdout=open('/dev/null','w'),stderr=subprocess.STDOUT)
    p.wait()

    end_time = time.time()
    print('Save mask end:', i, ',cost:', end_time - start_time)


if __name__ == '__main__':
    all_model_info = split_task()    

    # render
    render_start_time = time.time()
    process_array = []
    for i in range(NPROC):
        task_file = str(os.path.join(TASK_SPLIT_ROOT, '%d.txt' % i))
        print('render:', task_file)
        p = multiprocessing.Process(target=render_obj, args=(task_file,i))
        p.start()
        process_array.append(p)

    for i in range(NPROC):
        process_array[i].join()
    

    '''    
    pool = multiprocessing.Pool(processes = NPROC)
    for info in all_model_info:
        rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, info[0], info[1])
        if os.path.exists(os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % (N_VIEWS - 1))):
            continue
        pool.apply_async(render_single_obj, (info[0], info[1]) )
    pool.close()
    pool.join()
    '''
    
    render_end_time = time.time()
    print('Render all finished in %f sec' % (render_end_time - render_start_time))

    # transfer

    process_array = []
    transfer_start_time = time.time()
    
    for i in range(NPROC):
        task_file = str(os.path.join(TASK_SPLIT_ROOT, '%d.txt' % i))
        print('transfer:', task_file)
        p = multiprocessing.Process(target=get_rgb_depth_images, args=(task_file,i))
        p.start()
        process_array.append(p)

    for i in range(NPROC):
        process_array[i].join()
    
    '''
    pool = multiprocessing.Pool(processes = NPROC)
    for info in all_model_info:
        pool.apply_async(get_single_rgb_depth_images, (info[0], info[1]) )
    pool.close()
    pool.join()
    '''

    transfer_end_time = time.time()
    print('Transfer all finished in %f sec' % (transfer_end_time - transfer_start_time))

    # save mask

    process_array = []
    mask_start_time = time.time()

    for i in range(NPROC):
        task_file = str(os.path.join(TASK_SPLIT_ROOT, '%d.txt' % i))
        print('save mask:', task_file)
        p = multiprocessing.Process(target=get_mask, args=(task_file,i))
        p.start()
        process_array.append(p)

    for i in range(NPROC):
        process_array[i].join()

    mask_end_time = time.time()
    print('Save mask all finished in %f sec' % (mask_end_time - mask_start_time))    

    print('finished!')   
