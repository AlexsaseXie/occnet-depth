import numpy
import OpenEXR

DIR_RENDERING_PATH = '/home2/xieyunwei/occupancy_networks/data/render'

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
    depth_max = depth[depth < 5].max()
    depth[depth > 5] = depth_max
    #depth[im[:,:,3] == 0.] = (depth_min + depth_max) / 2
    depth = (depth - depth_min) / (depth_max - depth_min)
    return x, y, im, depth, depth_min, depth_max


import os
import sys
from PIL import Image
def main():
    model_id = ['2c981b96364b1baa21a66e8dfcce514a']
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
    main()
