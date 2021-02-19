import os
from PIL import Image, ImageDraw
import json
import numpy as np
from tqdm import tqdm
import argparse
import trimesh
import utils

parser = argparse.ArgumentParser('Test mask')
parser.add_argument('--test_num', type=int, default=10, help='Test num')
parser.add_argument('--pix3d_root', type=str, default='.')
parser.add_argument('--sampled_list_root', type=str, default='./sampled_list/')
parser.add_argument('--output_list_root', type=str, default='./generated_sampled_list/')
parser.add_argument('--generate_root', type=str, default='./test_img_generation/')

group1 = parser.add_mutually_exclusive_group()
group1.add_argument('--random_padding', action='store_true', default=False)
parser.add_argument('--padding_ratio_max', type=float, default=0.2)
parser.add_argument('--padding_ratio_min', type=float, default=0.05)
group1.add_argument('--fit_focal_length', action='store_true', default=False)

parser.add_argument('--target_focal_length', type=float, default=35)
parser.add_argument('--target_resolution', type=int, default=224)
parser.add_argument('--do_not_translate_trans_mat', action='store_true', default=False)

parser.add_argument('--intermediate_output', action='store_true', default=False)
parser.add_argument('--select_object_ratio', type=float, default=0.5)
parser.add_argument('--black', action='store_true') 
args = parser.parse_args()

TEST_CLASSES = [
    'sofa',
    'chair',
    'table'
]

PIX3D_ROOT = args.pix3d_root
SAMPLED_LIST_ROOT = args.sampled_list_root
GENERATE_ROOT = args.generate_root
TEST_NUM = args.test_num


test_class_infos = utils.read_sampled_list(SAMPLED_LIST_ROOT=SAMPLED_LIST_ROOT,TEST_CLASSES=TEST_CLASSES)

if not os.path.exists(GENERATE_ROOT):
    os.mkdir(GENERATE_ROOT)



def procedure_output(image_info, intermediate_output=True):
    Rt = utils.get_Rt(image_info)
    K = utils.get_K(image_info)

    # pre-process
    source_image = utils.get_image(image_info, pix3d_root=PIX3D_ROOT).convert('RGB')
    source_image_mask = utils.get_mask(image_info, pix3d_root=PIX3D_ROOT)
    if not args.black:
        background_image = Image.new('RGB', source_image.size, 'white')
    else:
        background_image = Image.new('RGB', source_image.size, 'black')
    source_image = Image.composite(source_image, background_image, source_image_mask)


    # Step 1 & 2: Translate the object area and change size to square images
    # calculate model center on the image coordinate system
    model_origin = np.array([[0.], [0.], [0.], [1.]])
    model_origin_on_image = np.dot(K, np.dot(Rt, model_origin))
    model_origin_on_image = model_origin_on_image[:2,0] / model_origin_on_image[2, 0]
    # transfer to int
    model_origin_on_image = model_origin_on_image.astype(np.int)
    image_size = np.array(utils.get_image_size(image_info)).astype(np.int)

    # the model_origin_on_image should be already near the center of the image
    #print('Image center:', image_size / 2.)
    #print('Model origin on image:', model_origin_on_image)

    object_bbox = np.array(utils.get_bbox(image_info))

    to_top_left = model_origin_on_image - object_bbox[:2]
    to_bottom_right = model_origin_on_image - object_bbox[2:4]

    x_axis_dis = max(model_origin_on_image[0] - object_bbox[0], object_bbox[2] - model_origin_on_image[0])
    y_axis_dis = max(model_origin_on_image[1] - object_bbox[1], object_bbox[3] - model_origin_on_image[1])
    max_axis = max(x_axis_dis, y_axis_dis)

    
    # determine the image size to change to
    if args.random_padding:
        # random padding
        padding_ratio = np.random.uniform(args.padding_ratio_min, args.padding_ratio_max)
        target_width = max_axis * 2 * (1 + padding_ratio)
        new_center = np.array([target_width / 2.] * 2, dtype=float) 
    elif args.fit_focal_length:
        # Find corresponding width at fixed focal length
        # Notice: 
        # This may result in very large images
        target_focal_length = args.target_focal_length
        target_width = K[0, 0] * 32. / target_focal_length
        new_center = np.array([target_width / 2.] * 2, dtype=float)
    else:
        # default: use source img width or img height
        target_width = max(image_size[0], image_size[1])
        new_center = np.array([target_width / 2.] * 2, dtype=float)

    # select
    if target_width <= 2 * max_axis:
        # object is too big
        return None
    if args.select_object_ratio != 0. and target_width * args.select_object_ratio > 2 * max_axis:
        # object is too small
        return None

    new_center = new_center.astype(np.int)
    del target_width
    new_image_size = new_center * 2

    # create new image
    if not args.black:
        img_after_translation = Image.new('RGB', (new_image_size[0], new_image_size[1]), 'white')
    else:
        img_after_translation = Image.new('RGB', (new_image_size[0], new_image_size[1]), 'black')
    mask_after_translation = Image.new('L', (new_image_size[0], new_image_size[1]), 'black')

    # crop & paste for source image
    object_pixels = source_image.crop(object_bbox)
    img_after_translation.paste(
        object_pixels, (
            new_center[0] - to_top_left[0],
            new_center[1] - to_top_left[1],
            new_center[0] - to_bottom_right[0],
            new_center[1] - to_bottom_right[1]
        )
    )

    # crop & paste for source image mask
    object_mask_pixels = source_image_mask.crop(object_bbox)
    mask_after_translation.paste(
        object_mask_pixels, (
            new_center[0] - to_top_left[0],
            new_center[1] - to_top_left[1],
            new_center[0] - to_bottom_right[0],
            new_center[1] - to_bottom_right[1]
        )
    )
    mask_after_translation = mask_after_translation.point(lambda x: x >= 128, '1')


    # save img after step 1,2 
    image_name = utils.get_image_name(image_info)
    image_generate_root = os.path.join(cls_generate_root, image_name)
    if not os.path.exists(image_generate_root):
        os.mkdir(image_generate_root)

    if intermediate_output:
        img_after_translation.save(os.path.join(image_generate_root, '%s_after_translation.png' % image_name))
        mask_after_translation.save(os.path.join(image_generate_root, '%s_after_translation_mask.png' % image_name))

    

    if args.do_not_translate_trans_mat:
        # method 1: only modify the K mat by translation
        # change size does not matter
        translation_vector = new_center - model_origin_on_image
        Rt_after_tranlation = Rt.copy()
        K_after_translation = K.copy()
        # tranlation on image coordinate system
        K_after_translation[0, 2] = K[0, 2] + float(translation_vector[0])
        K_after_translation[1, 2] = K[1, 2] + float(translation_vector[1])  
    else:
        # method 2: modify K mat & trans mat together
        # change size does not matter
        translation_vector = new_center - image_size / 2.
        Rt_after_tranlation = Rt.copy()
        K_after_translation = K.copy()

        # reset model translation
        Rt_after_tranlation[0, 3] = 0.
        Rt_after_tranlation[1, 3] = 0.
        # tranlation on image coordinate system
        K_after_translation[0, 2] = K[0, 2] + float(translation_vector[0])
        K_after_translation[1, 2] = K[1, 2] + float(translation_vector[1])


    # create 3d keypoint image
    if intermediate_output:
        three_d_keypoints = utils.get_3d_keypoints(image_info, pix3d_root=PIX3D_ROOT)
        three_d_keypoints_img_after_translation = img_after_translation.copy()

        K_dot_Rt_after_translation = np.dot(K_after_translation, Rt_after_tranlation)
        three_d_keypoints_img_after_translation = utils.draw_3d_keypoints(image_info, 
            three_d_keypoints_img_after_translation, three_d_keypoints, 
            DRAW_SIZE=0.01 * three_d_keypoints_img_after_translation.size[0], 
            K_dot_Rt= K_dot_Rt_after_translation
        )

        if args.do_not_translate_trans_mat:
            three_d_keypoints_img_after_translation.save(
                os.path.join(image_generate_root, '%s_after_translation_3d_keypoints.png' % image_name)
            )
        else:
            three_d_keypoints_img_after_translation.save(
                os.path.join(image_generate_root, '%s_after_translation_3d_keypoints_modify_trans_mat.png' % image_name)
            )
    
    # Step 3: Resize to 224
    final_image_size = [args.target_resolution] * 2
    img_final = img_after_translation.resize(final_image_size, Image.ANTIALIAS)
    mask_final = mask_after_translation.resize(final_image_size, Image.ANTIALIAS)
    
    img_final.save(os.path.join(image_generate_root, '%s_final.png' % image_name))
    mask_final.save(os.path.join(image_generate_root, '%s_final_mask.png' % image_name))

    # create 3d keypoint image
    # calculate new K
    ratio = final_image_size[0] / new_image_size[0]
    K_final = K_after_translation.copy()
    K_final[0,:] = K_final[0,:] * ratio
    K_final[1,:] = K_final[1,:] * ratio

    # create 3d keypoint image
    if intermediate_output:
        K_dot_Rt_final = np.dot(K_final, Rt_after_tranlation)
        three_d_keypoints_img_final = img_final.copy()

        three_d_keypoints_img_final = utils.draw_3d_keypoints(image_info, 
            three_d_keypoints_img_final, three_d_keypoints, 
            DRAW_SIZE=0.01 * three_d_keypoints_img_final.size[0], 
            K_dot_Rt= K_dot_Rt_final
        )

        three_d_keypoints_img_final.save(
            os.path.join(image_generate_root, '%s_final_3d_keypoints.png' % image_name)
        )

    final_focal_length = K_final[0,0] * 32. / final_image_size[0]
    #print('Final focal length:', final_focal_length)

    return_dict = {
        'img_size_%d' % args.target_resolution : args.target_resolution,
        'K_%d' % args.target_resolution     : K_final.tolist(),
        'Rt_%d' % args.target_resolution    : Rt_after_tranlation.tolist(),
        'img_%d' % args.target_resolution   : os.path.abspath(os.path.join(image_generate_root, '%s_final.png' % image_name)),
        'mask_%d' % args.target_resolution  : os.path.abspath(os.path.join(image_generate_root, '%s_final_mask.png' % image_name)),
        'focal_length_%d' % args.target_resolution : final_focal_length
    }

    return return_dict

OUTPUT_SAMPLED_LIST_ROOT = args.output_list_root
if not os.path.exists(OUTPUT_SAMPLED_LIST_ROOT):
    os.mkdir(OUTPUT_SAMPLED_LIST_ROOT)


output_cls_infos = {
    'sofa': [],
    'chair': [],
    'table': []
}

for test_cls in TEST_CLASSES:
    cls_infos = test_class_infos[test_cls]

    cls_generate_root = os.path.join(GENERATE_ROOT, test_cls)
    if not os.path.exists(cls_generate_root):
        os.mkdir(cls_generate_root)

    if TEST_NUM != -1:
        choices = range(TEST_NUM)
    else:
        choices = range(len(cls_infos))

    for index in tqdm(choices):
        image_info = cls_infos[index]

        return_dict = procedure_output(image_info, intermediate_output=args.intermediate_output)

        if return_dict is not None:
            image_info.update(return_dict)

            # calculate model scale & traslation
            model_path = os.path.join(PIX3D_ROOT, image_info['model'])
            mesh = trimesh.load(model_path, process=False)
            bbox = mesh.bounding_box.bounds
            loc = (bbox[0] + bbox[1]) / 2.
            scale = (bbox[1] - bbox[0]).max() / 1.

            image_info['model_loc'] = loc.tolist()
            image_info['model_scale'] = scale

            output_cls_infos[test_cls].append(image_info)

    print('class %s has %d valid images' % (test_cls, len(output_cls_infos[test_cls])) )

    cls_sampling_root = os.path.join(OUTPUT_SAMPLED_LIST_ROOT, test_cls)

    if not os.path.exists(cls_sampling_root):
        os.mkdir(cls_sampling_root)
    with open(os.path.join(cls_sampling_root, 'all_info.json'), 'w') as f:
        json.dump(output_cls_infos[test_cls], f)

        


