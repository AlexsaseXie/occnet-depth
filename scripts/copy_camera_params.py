import os
import shutil
import numpy as np
from tqdm import tqdm

DATASET_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.with_depth.10w10w/'

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

for c in CLASSES:
    onet_class_root = os.path.join(DATASET_PATH, c)
    modelnames = os.listdir(onet_class_root)


    print('CLASS %s:%d' % (c, len(modelnames)))    
    for modelname in modelnames:
        model_root = os.path.join(onet_class_root, modelname)

        camera_npz_root = os.path.join(model_root, 'img', 'cameras.npz')
        points_npz_root = os.path.join(model_root, 'points.npz')
        if os.path.exists(camera_npz_root):
            data = np.load(camera_npz_root)
            data = dict(data)
            points_data = np.load(points_npz_root)

            data['loc'] = points_data['loc']
            data['scale'] = points_data['scale']

            np.savez(camera_npz_root, **data)

