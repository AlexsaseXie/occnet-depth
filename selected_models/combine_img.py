from PIL import Image
import os
from tqdm import tqdm
import argparse
import re
import random


SELECTED_MODEL_ROOT = '/home2/xieyunwei/occupancy_networks/selected_models/selected_models/'
RENDERING_DIR_NAME = 'renderings'
FILES = [
        'gt_rgb.png', 
        '%s/atlasnet.png' % RENDERING_DIR_NAME, 
        '%s/pix2mesh.png' % RENDERING_DIR_NAME,
        '%s/disn.png' % RENDERING_DIR_NAME, 
        '%s/imnet.png' % RENDERING_DIR_NAME, 
        '%s/onet.png' % RENDERING_DIR_NAME, 
        '%s/ours.png' % RENDERING_DIR_NAME,
        '%s/gt.png' % RENDERING_DIR_NAME,  
    ]
RENDERING_RESOLUTION = 500
OUTPUT_FILE = 'sticker.png'
FINAL_OUTPUT_FILE = 'final_sticker_%d.png'

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a) 

def main():
    for class_name in os.listdir(SELECTED_MODEL_ROOT):
        class_root = os.path.join(SELECTED_MODEL_ROOT, class_name)

        if not os.path.isdir(class_root):
            continue

        for modelname in tqdm(os.listdir(class_root)):
            model_root = os.path.join(class_root, modelname)

            test_path = os.path.join(model_root, FILES[0])
            if not os.path.exists(test_path):
                continue

            sticker_count = len(FILES)
            new_size = (RENDERING_RESOLUTION * sticker_count, RENDERING_RESOLUTION)
            new_image = Image.new('RGBA', new_size)

            for idx, f in enumerate(FILES):
                f_path = os.path.join(model_root, f)
                image = Image.open(f_path)

                if image.size[0] != RENDERING_RESOLUTION:
                    image = image.resize((RENDERING_RESOLUTION, RENDERING_RESOLUTION), resample=Image.BILINEAR)

                new_image.paste(image, (idx * RENDERING_RESOLUTION, 0))

            output_file = os.path.join(model_root, OUTPUT_FILE)
            new_image.save(output_file)

def vertical_main(sticker_list, out_dir='final_stickers', vertical_sticker_count=13, shuffle=False):
    if not os.path.exists(sticker_list):
        return

    out_dir = os.path.join(SELECTED_MODEL_ROOT, out_dir)
    mkdir_p(out_dir)

    with open(sticker_list, 'r') as f:
        model_infos = f.readlines()

    model_info_list = []
    for model_info in model_infos:
        tmp = model_info.strip().split('\t')
        if len(model_info) <= 5 or tmp[0].startswith('class'):
            continue

        model_info_list.append([tmp[0], tmp[1]])

    if shuffle:
        random.shuffle(model_info_list)

    total_sticker_num = len(model_info_list) // vertical_sticker_count
    if len(model_info_list) % vertical_sticker_count != 0:
        total_sticker_num += 1

    for i in tqdm(range(total_sticker_num)):
        if i != total_sticker_num - 1:
            cur_vertical_sticker_num = vertical_sticker_count
        else:
            cur_vertical_sticker_num = len(model_info_list) - vertical_sticker_count * i 
        new_size = (RENDERING_RESOLUTION * len(FILES), RENDERING_RESOLUTION * cur_vertical_sticker_num)
        new_image = Image.new('RGBA', new_size)

        for k in range(cur_vertical_sticker_num):
            index = vertical_sticker_count * i + k

            class_id = model_info_list[index][0]
            modelname = model_info_list[index][1]

            raw_sticker_path = os.path.join(SELECTED_MODEL_ROOT, class_id, modelname, OUTPUT_FILE)
            img = Image.open(raw_sticker_path)

            new_image.paste(img, (0, k * RENDERING_RESOLUTION))
        
        output_path = os.path.join(out_dir, FINAL_OUTPUT_FILE % i)

        new_image = new_image.resize((new_image.size[0] // 2, new_image.size[1] // 2) , resample=Image.BILINEAR)
        new_image.save(output_path)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Select Model'
    )
    parser.add_argument('--vertical', action='store_true', help='Vectical stick images') 
    parser.add_argument('--vertical_sticker_file', type=str, default='selected_best.txt', help='Vertical sticker model names')
    parser.add_argument('--vertical_sticker_output_dir', type=str, default='final_stickers', help='output dir')
    parser.add_argument('--vertical_sticker_count', type=int, default=13)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()

    if not args.vertical:
        main()
    else:
        vertical_main(
            args.vertical_sticker_file, 
            out_dir=args.vertical_sticker_output_dir, 
            vertical_sticker_count=args.vertical_sticker_count,
            shuffle=args.shuffle
        )

        

    
