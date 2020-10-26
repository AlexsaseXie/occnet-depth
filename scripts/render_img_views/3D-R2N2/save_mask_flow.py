#from ../../../im2mesh.utils.mask_flow import mask_flow
import numpy as np
from rendering_config import *

import os
import sys
sys.path.append('../../../')
from im2mesh.utils.mask_flow import mask_flow

import time
from PIL import Image
import argparse

def main(args):
    all_model_class = []
    all_model_ids = []

    with open(args.task_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line == '': 
                continue

            tmp = line.rstrip('\n').split(' ')
            all_model_class.append(tmp[0])
            all_model_ids.append(tmp[1])

    for i, curr_model_id in enumerate(all_model_ids):
        start_time = time.time()
        rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, all_model_class[i], all_model_ids[i])
        rendering_curr_model_save_mask_root = os.path.join(rendering_curr_model_root, 'rendering_mask')
        rendering_curr_model_save_mask_flow_root = os.path.join(rendering_curr_model_root, 'rendering_mask_flow')
        
        assert os.path.exists(rendering_curr_model_save_mask_root)
        if not os.path.exists(rendering_curr_model_save_mask_flow_root):
            os.mkdir(rendering_curr_model_save_mask_flow_root)

        if os.path.exists(os.path.join(rendering_curr_model_save_mask_flow_root, '%.2d_mask_flow.png' % (N_VIEWS - 1))):
            continue
        
        for view_id in range(N_VIEWS):
            mask_path = os.path.join(rendering_curr_model_save_mask_root, '%.2d_mask.png' % view_id)
            
            depth_mask = Image.open(mask_path).convert('1')
            depth_mask_array = np.array(depth_mask)
            
            mask_flow_array = mask_flow(depth_mask_array)
            
            mask_flow_img = Image.fromarray((mask_flow_array * 255.).astype(np.uint8), 'L')
            mask_flow_img.save(os.path.join(rendering_curr_model_save_mask_flow_root, '%.2d_mask_flow.png' % view_id))     

        end_time = time.time()
        print('transfer model in', end_time - start_time, ' secs')

def main_single(args):
    rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, args.model_class, args.model_id)
    rendering_curr_model_save_mask_root = os.path.join(rendering_curr_model_root, 'rendering_mask')
    rendering_curr_model_save_mask_flow_root = os.path.join(rendering_curr_model_root, 'rendering_mask_flow')
        
    assert os.path.exists(rendering_curr_model_save_mask_root)
    if not os.path.exists(rendering_curr_model_save_mask_flow_root):
        os.mkdir(rendering_curr_model_save_mask_flow_root)

    if os.path.exists(os.path.join(rendering_curr_model_save_mask_flow_root, '%.2d_mask_flow.png' % (N_VIEWS - 1))):
        return
    
    for view_id in range(N_VIEWS):
        mask_path = os.path.join(rendering_curr_model_save_mask_root, '%.2d_mask.png' % view_id)
        
        depth_mask = Image.open(mask_path).convert('1')
        depth_mask_array = np.array(depth_mask)
        
        mask_flow_array = mask_flow(depth_mask_array)
        
        mask_flow_img = Image.fromarray((mask_flow_array * 255.).astype(np.uint8), 'L')
        mask_flow_img.save(os.path.join(rendering_curr_model_save_mask_flow_root, '%.2d_mask_flow.png' % view_id))


def test():
    n_view = 0
    model_class = ['02691156']
    model_id = ['10155655850468db78d106ce0a280f87']
    for i, curr_model_id in enumerate(model_id):
        mask_root = os.path.join(DIR_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask')
        mask_path = os.path.join(mask_root, '%.2d_mask.png' % n_view)

        depth_mask = Image.open(mask_path).convert('1')

        # mkdirs
        if not os.path.exists(TEST_RENDERING_PATH):
            os.mkdir(TEST_RENDERING_PATH)        

        if not os.path.exists(os.path.join(TEST_RENDERING_PATH, model_class[i])):
            os.mkdir(os.path.join(TEST_RENDERING_PATH, model_class[i]))

        if not os.path.exists(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id)):
            os.mkdir(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id))

        if not os.path.exists(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask')):
            os.mkdir(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask'))

        if not os.path.exists(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask_flow')):
            os.mkdir(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask_flow'))

        # save input
        depth_mask.save(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask', '%.2d_mask.png' % n_view))
        depth_mask_array = np.array(depth_mask)
        
        print('depth_mask_array:',depth_mask_array.shape)

        mask_flow_array, borders_array = mask_flow(depth_mask_array, True)
        print(borders_array)
             
        borders_mask = Image.fromarray(borders_array.astype(np.uint8))
        borders_mask = borders_mask.point(lambda i: i == 1, '1')
        borders_mask.save(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask_flow', '%.2d_mask.png' % n_view))

        mask_flow_img = Image.fromarray(mask_flow_array, 'F')
        mask_flow_img.save(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask_flow', '%.2d_mask_flow.tiff' % n_view))
        mask_flow_img = Image.fromarray((mask_flow_array * 255.).astype(np.uint8), 'L')
        mask_flow_img.save(os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_mask_flow', '%.2d_mask_flow.png' % n_view))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert exr to pngs')
    parser.add_argument('--task_file', type=str, help='task split file')
    parser.add_argument('--single', action='store_true', help='use single')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--model_class', type=str, default='', help='model class')
    parser.add_argument('--model_id', type=str, default='', help='model id')
    args = parser.parse_args()

    if args.test:
        test()
        exit(0)

    if not args.single:
        main(args)
    else:
        main_single(args)
