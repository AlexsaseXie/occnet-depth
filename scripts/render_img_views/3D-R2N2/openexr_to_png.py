import numpy
import OpenEXR

DIR_RENDERING_PATH = '/home2/xieyunwei/occupancy_networks/data/render_2'
N_VIEWS = 24
RENDERING_MAX_CAMERA_DIST = 1.75

def norm(val):
    return val * 12.92 if val <= 0.0031308 else 1.055 * val**(1.0/2.4) - 0.055
norm = numpy.vectorize(norm)

def get_exr_dim(image):
    header = image.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    return size

def convert_OpenEXR_to_sRGB(path):
    image = OpenEXR.InputFile(path)
    x, y = get_exr_dim(image)
    im = numpy.zeros((x,y,4))

    im[:,:,0] = norm(numpy.frombuffer(image.channel('R'), dtype=numpy.float32).reshape((x,y)))
    im[:,:,1] = norm(numpy.frombuffer(image.channel('G'), dtype=numpy.float32).reshape((x,y)))
    im[:,:,2] = norm(numpy.frombuffer(image.channel('B'), dtype=numpy.float32).reshape((x,y)))
    im[:,:,3] = numpy.frombuffer(image.channel('A'), dtype=numpy.float32).reshape((x,y))

    im = numpy.clip(im, 0, 1)
    
    depth = numpy.zeros((x,y))
    depth[:,:] = numpy.frombuffer(image.channel('Z'), dtype=numpy.float32).reshape((x,y))
    depth_min = depth.min()
    depth_max = depth[depth < 10].max()
    depth[depth >= 10] = depth_max
    #depth[im[:,:,3] == 0.] = (depth_min + depth_max) / 2
    depth = (depth - depth_min) / (depth_max - depth_min)
    return x, y, im, depth, depth_min, depth_max


import os
import sys
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
        rendering_curr_model_save_png_root = os.path.join(rendering_curr_model_root, 'rendering_png')
        if not os.path.exists(rendering_curr_model_save_png_root):
            os.mkdir(rendering_curr_model_save_png_root)

        if os.path.exists(os.path.join(rendering_curr_model_save_png_root, '%.2d_rgb.png' % (N_VIEWS - 1))):
            continue
        
        rf = open(os.path.join(rendering_curr_model_root, 'rendering_metadata.txt'), 'r')
        f = open(os.path.join(rendering_curr_model_save_png_root, 'depth_range.txt'), 'w')
        for view_id in range(N_VIEWS):
            image_path = os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % view_id)
            
            try:
                x, y, img, depth, depth_min, depth_max = convert_OpenEXR_to_sRGB(image_path)
            except:
                continue
            finally:
                pass
            
            img = (img * 255.0).astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(rendering_curr_model_save_png_root, '%.2d_rgba.png' % view_id))

            depth = (depth * 255.).astype(numpy.uint8)
            depth = Image.fromarray(depth)
            depth = depth.convert('L')
            # save depth map & range
            depth.save(os.path.join(rendering_curr_model_save_png_root, '%.2d_depth.png' % view_id))
                     
            depth_unit = RENDERING_MAX_CAMERA_DIST * float(rf.readline().split(' ')[3])
            print(depth_min, depth_max, depth_unit, file=f)
            
            # convert to jpg, save jpg
            background = Image.new('RGBA', (x,y), (255,255,255,255))
            img_rgb = Image.alpha_composite(background, img)
            img_rgb.convert('RGB').save(os.path.join(rendering_curr_model_save_png_root, '%.2d_rgb.png' % view_id))
            #print('depth min:', depth_min, ',max:', depth_max)
            
        f.close()
        rf.close()
        end_time = time.time()
        print('transfer model in', end_time - start_time, ' secs')

def main_single(args):
    rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, args.model_class, args.model_id)
    rendering_curr_model_save_png_root = os.path.join(rendering_curr_model_root, 'rendering_png')
    if not os.path.exists(rendering_curr_model_save_png_root):
        os.mkdir(rendering_curr_model_save_png_root)

    if os.path.exists(os.path.join(rendering_curr_model_save_png_root, '%.2d_rgb.png' % (N_VIEWS - 1))):
        return

    rf = open(os.path.join(rendering_curr_model_root, 'rendering_metadata.txt'), 'r')
    f = open(os.path.join(rendering_curr_model_save_png_root, 'depth_range.txt'), 'w')
    for view_id in range(N_VIEWS):
        image_path = os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % view_id)
        try:
            x, y, img, depth, depth_min, depth_max = convert_OpenEXR_to_sRGB(image_path)
        except:
            continue
        finally:
            pass
        
        img = (img * 255.0).astype(numpy.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(rendering_curr_model_save_png_root, '%.2d_rgba.png' % view_id))

        depth = (depth * 255.).astype(numpy.uint8)
        depth = Image.fromarray(depth)
        depth = depth.convert('L')
        # save depth map & range
        depth.save(os.path.join(rendering_curr_model_save_png_root, '%.2d_depth.png' % view_id))
        depth_unit = RENDERING_MAX_CAMERA_DIST * float(rf.readline().split(' ')[3])
        print(depth_min, depth_max, depth_unit, file=f)

        # convert to jpg, save jpg
        background = Image.new('RGBA', (x,y), (255,255,255,255))
        img_rgb = Image.alpha_composite(background, img)
        img_rgb.convert('RGB').save(os.path.join(rendering_curr_model_save_png_root, '%.2d_rgb.png' % view_id))
        #print('depth min:', depth_min, ',max:', depth_max)

    f.close()
    rf.close()

def test():
    model_class = ['04090263']
    model_id = ['4a32519f44dc84aabafe26e2eb69ebf4']
    for i, curr_model_id in enumerate(model_id):
        image_path = '%s/%s.exr' % (DIR_RENDERING_PATH, curr_model_id)

        x, y, img, depth, depth_min, depth_max = convert_OpenEXR_to_sRGB(image_path)
        img = (img * 255.0).astype(numpy.uint8)

        img = Image.fromarray(img)
        img.save(image_path[:-4] + '_converted.png')

        depth = (depth * 255.).astype(numpy.uint8)
        depth = Image.fromarray(depth)
        depth = depth.convert('L')
        depth.save(image_path[:-4] + '_converted_depth.png')

        # convert to jpg, save jpg
        background = Image.new('RGBA', (x,y), (255,255,255,255))
        img_rgb = Image.alpha_composite(background, img)
        img_rgb.convert('RGB').save(image_path[:-4] + '_converted_rgb.png')
        print('depth min:', depth_min, ',max:', depth_max)

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
