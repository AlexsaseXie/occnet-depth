import os
import sys
import time
import datetime
import numpy as np
from tqdm import tqdm

SHAPENET_ROOT = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/'
R2N2_ROOT = '/home2/xieyunwei/occupancy_networks/external/Choy2016/ShapeNetRendering/'
DATASET_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.with_depth.10w10w'
#DATASET_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.depth_pred.uresnet.origin_subdivision/'

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

N_VIEWS = 24

def main():
    print('Counting:')
    for model_class in CLASSES:
        missing_count = 0

        class_root = os.path.join(SHAPENET_ROOT, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_root)

        for model_id in current_class_ids:
            rendering_curr_model_root = os.path.join(DATASET_PATH, model_class, model_id)
            if not os.path.exists(os.path.join(rendering_curr_model_root, 'depth', 'depth_range.txt')):
                #print('%s/%s is missing' % (model_class, model_id))
                missing_count += 1
                continue
            
            #for i in range(24):
            #    if not os.path.exists(os.path.join(rendering_curr_model_root, 'depth_pointcloud', '%.2d_pointcloud.npz'%i)):
                    #print('%s/%s is missing' % (model_class, model_id))
            #        missing_count += 1
            #        break
            
            '''
            f_path = os.path.join(rendering_curr_model_root, 'depth', 'depth_range.txt')
            file_time = time.strftime("%Y%m%d %H:%M:%S",time.localtime(os.stat(f_path).st_mtime))
            local_time = time.strftime("%Y%m%d %H:%M:%S",time.localtime(time.time()))
            starttime = datetime.datetime.strptime(file_time,"%Y%m%d %H:%M:%S")
            endtime = datetime.datetime.strptime(local_time,"%Y%m%d %H:%M:%S")
            if (endtime-starttime).seconds > 3600:
                missing_count += 1
            '''

        print('Class %s missing: %d / %d' % (model_class, missing_count, len(current_class_ids)))

def fix_dataset():
    print('Counting:')
    for model_class in CLASSES:
        missing_count = 0

        class_dataset_root = os.path.join(DATASET_PATH, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_dataset_root)

        for model_id in tqdm(current_class_ids):
            model_root = os.path.join(class_dataset_root, model_id)
            model_points_file = os.path.join(model_root, 'points_direct.npz')
            model_pointcloud_file = os.path.join(model_root, 'pointcloud_direct.npz')
            if os.path.exists(model_points_file):
                #print('%s/%s fixed' % (model_class, model_id))
                point_dict = np.load(model_points_file)
                true_scale = point_dict['scale']
                true_loc = point_dict['loc']

                pointcloud_data = np.load(model_pointcloud_file)
                pointcloud = pointcloud_data['points']
                normals = pointcloud_data['normals']
                np.savez(model_pointcloud_file, points=pointcloud, 
                    normals=normals, loc=true_loc, scale=true_scale)

def fix2_dataset():
    print('Fix tsdf:')
    for model_class in CLASSES:
        missing_count = 0

        class_dataset_root = os.path.join(DATASET_PATH, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_dataset_root)

        for model_id in tqdm(current_class_ids):
            model_root = os.path.join(class_dataset_root, model_id)
            model_points_file = os.path.join(model_root, 'points_direct_tsdf0.008.npz')
            if os.path.exists(model_points_file):
                #print('%s/%s fixed' % (model_class, model_id))
                point_dict = np.load(model_points_file)
                tsdf = point_dict['tsdf']

                nan_count = np.isnan(tsdf).sum()
                if nan_count > 0:
                    print('Has nan: %d in %s' % (nan_count,model_points_file) )

if __name__ == '__main__':
    #main()
    fix2_dataset()

