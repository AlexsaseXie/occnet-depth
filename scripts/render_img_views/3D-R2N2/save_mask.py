import numpy
import OpenEXR
from rendering_config import *
import os
import sys
import time
from PIL import Image
import argparse


def get_exr_dim(image):
    header = image.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    return size

def get_mask(path):
    image = OpenEXR.InputFile(path)
    x, y = get_exr_dim(image)
    
    depth = numpy.zeros((x,y))
    depth[:,:] = numpy.frombuffer(image.channel('Z'), dtype=numpy.float32).reshape((x,y))
    mask = (depth < 10).astype(numpy.uint8)
    return x, y, mask

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a)

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
        if not os.path.exists(rendering_curr_model_save_mask_root):
            os.mkdir(rendering_curr_model_save_mask_root)

        if os.path.exists(os.path.join(rendering_curr_model_save_mask_root, '%.2d_mask.png' % (N_VIEWS - 1))):
            continue
        
        for view_id in range(N_VIEWS):
            image_path = os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % view_id)
            
            try:
                x, y, mask = get_mask(image_path)
            except:
                continue
            finally:
                pass
            
            mask = Image.fromarray(mask)
            mask = mask.point(lambda i: i == 1, '1')
            # save mask
            mask.save(os.path.join(rendering_curr_model_save_mask_root, '%.2d_mask.png' % view_id))
                     
        end_time = time.time()
        print('transfer model in', end_time - start_time, ' secs')

def main_single(args):
    rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, args.model_class, args.model_id)
    rendering_curr_model_save_mask_root = os.path.join(rendering_curr_model_root, 'rendering_mask')
    if not os.path.exists(rendering_curr_model_save_mask_root):
        os.mkdir(rendering_curr_model_save_mask_root)

    if os.path.exists(os.path.join(rendering_curr_model_save_mask_root, '%.2d_mask.png' % (N_VIEWS - 1))):
        return

    for view_id in range(N_VIEWS):
        image_path = os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % view_id)
            
        try:
            x, y, mask = get_mask(image_path)
        except:
            continue
        finally:
            pass
        
        mask = Image.fromarray(mask)
        mask = mask.point(lambda i: i == 1, '1')
        # save mask
        mask.save(os.path.join(rendering_curr_model_save_mask_root, '%.2d_mask.png' % view_id))

def test():
    model_class = TEST_MODEL_CLASSES
    model_id = TEST_MODEL_IDS
    for i, curr_model_id in enumerate(model_id):
        for view_id in range(N_VIEWS):
            image_path = os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_exr', '%.2d.exr' % view_id)

            assert os.path.exists(image_path)

            save_root = os.path.join(TEST_RENDERING_PATH, model_class[i], model_id[i], 'rendering_mask')
            mkdir_p(save_root)

            x, y, mask = get_mask(image_path)
            
            #print(mask.shape, mask.dtype)
            mask = Image.fromarray(mask)
            mask = mask.point(lambda i: i == 1, '1')
            # save mask
            mask_path = os.path.join(save_root, '%.2d_mask.png' % view_id) 
            mask.save(mask_path)

            # method 2
            image_path = os.path.join(TEST_RENDERING_PATH, model_class[i], curr_model_id, 'rendering_png', '%.2d_rgb.png' % view_id)

            img = Image.open(image_path)
            img = numpy.array(img)

            mask = (img != 0).astype(numpy.uint8)
            mask = mask.sum(2).astype(numpy.uint8)
            #print(mask.shape, mask.dtype)
            mask = Image.fromarray(mask)
            mask = mask.point(lambda i: i > 0, '1')
            # save mask
            mask_path = os.path.join(save_root, '%.2d_mask2.png' % view_id)
            mask.save(mask_path)

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
