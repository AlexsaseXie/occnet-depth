import os
import numpy as np
from PIL import Image, ImageDraw
import trimesh
import json

# read sampled list
def read_sampled_list(SAMPLED_LIST_ROOT, TEST_CLASSES):
    test_class_infos = {}
    for test_cls in TEST_CLASSES:
        test_class_infos[test_cls] = []
        assert os.path.exists(SAMPLED_LIST_ROOT)
        with open(os.path.join(SAMPLED_LIST_ROOT, test_cls, 'all_info.json'), 'r') as f:
            test_class_infos[test_cls] = json.load(f)

    return test_class_infos



def get_Rt(image_info, pix3d_root=None):
    R = np.array(image_info['rot_mat'])
    t = np.resize(np.array(image_info['trans_mat']), [3,1])

    Rt = np.concatenate([R, t], axis=1)

    reverse_xy = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    return np.dot(reverse_xy, Rt)

def get_K(image_info, pix3d_root=None):
    estimated_focal_length = image_info['focal_length']
    focal_div_sensor = estimated_focal_length / 32.
    image_size = image_info['img_size']

    K = np.array([
        [focal_div_sensor * image_size[0], 0., image_size[0] / 2.],
        [0., focal_div_sensor * image_size[0], image_size[1] / 2.],
        [0., 0., 1.]
    ])

    return K

def get_image_size(image_info, pix3d_root=None):
    return image_info['img_size']

def get_image_name(image_info, pix3d_root=None):
    image_name = image_info['img'].split('/')[-1].split('.')[0]
    return image_name

def get_image(image_info, pix3d_root='.'):
    image_path = os.path.join(pix3d_root, image_info['img'])
    image = Image.open(image_path)
    return image

def get_2d_keypoints(image_info, pix3d_root=None):
    # get keypoints
    two_d_keypoints = np.array(image_info['2d_keypoints'])
    
    # combine all annotations
    if False:
        num_keypoints = two_d_keypoints.shape[1]
        remove_ = (two_d_keypoints[:,:,:] < 0).any(axis=(0,2))
        #print('remove_:', remove_, remove_.shape)
        two_d_keypoints = np.mean(two_d_keypoints, axis=0)
        two_d_keypoints[remove_ , :] = -1.
    else:
        two_d_keypoints = two_d_keypoints[0]
    
    return two_d_keypoints

def get_3d_keypoints(image_info, pix3d_root='.'):
    three_d_keypoints = image_info['3d_keypoints']

    with open(os.path.join(pix3d_root, three_d_keypoints)) as f:
        tmps = f.readlines()

    three_d_keypoints = []
    for tmp in tmps:
        if len(tmp) <= 3: 
            continue
        tmp = tmp.strip().split(' ')
        d_tmp = []
        for tmp_str in tmp:
            d_tmp.append(float(tmp_str))
        three_d_keypoints.append(np.array(d_tmp))

    return three_d_keypoints

# draw keypoints related operations
def draw_both_keypoints(image_info, 
    two_d_origin_img, three_d_origin_img, 
    two_d_keypoints, three_d_keypoints, 
    DRAW_SIZE=4, K_dot_Rt=None):
    # inplace operation

    if K_dot_Rt is None:
        Rt = get_Rt(image_info)
        K = get_K(image_info)
        K_dot_Rt = np.dot(K, Rt)

    two_d_draw = ImageDraw.Draw(two_d_origin_img)
    three_d_draw = ImageDraw.Draw(three_d_origin_img)

    for keypoint_i, keypoint in enumerate(two_d_keypoints):
        if keypoint[0] < 0. or keypoint[1] < 0.:
            continue

        # 2d keypoint
        two_d_draw.ellipse(
            (keypoint[0] - DRAW_SIZE, keypoint[1] - DRAW_SIZE, keypoint[0] + DRAW_SIZE, keypoint[1] + DRAW_SIZE),
            'red'
        )

        # 3d keypoint
        three_d_keypoint = three_d_keypoints[keypoint_i]
        p_3d = np.array([ [three_d_keypoint[0]], [three_d_keypoint[1]], [three_d_keypoint[2]], [1.] ])

        p_2d = np.dot(K_dot_Rt, p_3d)
        p_2d = p_2d[:2, 0] / p_2d[2, 0]

        three_d_draw.ellipse(
            (p_2d[0] - DRAW_SIZE, p_2d[1] - DRAW_SIZE, p_2d[0] + DRAW_SIZE, p_2d[1] + DRAW_SIZE),
            'blue'
        )

    return two_d_origin_img, three_d_origin_img

def draw_2d_keypoints(image_info, 
    two_d_origin_img,  
    two_d_keypoints, 
    DRAW_SIZE=4):
    # inplace operation

    two_d_draw = ImageDraw.Draw(two_d_origin_img)

    for keypoint in two_d_keypoints:
        if keypoint[0] < 0. or keypoint[1] < 0.:
            continue

        # 2d keypoint
        two_d_draw.ellipse(
            (keypoint[0] - DRAW_SIZE, keypoint[1] - DRAW_SIZE, keypoint[0] + DRAW_SIZE, keypoint[1] + DRAW_SIZE),
            'red'
        )

    return two_d_origin_img

def draw_3d_keypoints(image_info, 
    three_d_origin_img, 
    three_d_keypoints, 
    DRAW_SIZE=4, K_dot_Rt=None):
    # inplace operation

    if K_dot_Rt is None:
        Rt = get_Rt(image_info)
        K = get_K(image_info)
        K_dot_Rt = np.dot(K, Rt)

    three_d_draw = ImageDraw.Draw(three_d_origin_img)

    for three_d_keypoint in three_d_keypoints:
        # 3d keypoint
        p_3d = np.array([ [three_d_keypoint[0]], [three_d_keypoint[1]], [three_d_keypoint[2]], [1.] ])

        p_2d = np.dot(K_dot_Rt, p_3d)
        p_2d = p_2d[:2, 0] / p_2d[2, 0]

        three_d_draw.ellipse(
            (p_2d[0] - DRAW_SIZE, p_2d[1] - DRAW_SIZE, p_2d[0] + DRAW_SIZE, p_2d[1] + DRAW_SIZE),
            'blue'
        )

    return three_d_origin_img

def get_mask(image_info, pix3d_root='.'):
    mask_path = os.path.join(pix3d_root, image_info['mask'])

    mask_img = Image.open(mask_path) # mode:L

    return mask_img

def get_model(image_info, pix3d_root='.'):
    model_path = os.path.join(pix3d_root, image_info['model'])
    mesh = trimesh.load(model_path, process=False)
    return mesh

def get_bbox(image_info, pix3d_root=None):
    return image_info['bbox']

def get_model_name(image_info, pix3d_root='.'):
    model_path = os.path.join(pix3d_root, image_info['model'])
    return model_path.split('/')[-2]