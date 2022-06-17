import os
import sys
import time
import datetime
import trimesh
import numpy as np
from tqdm import tqdm
import trimesh

SHAPENET_ROOT = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/'
R2N2_ROOT = '/home2/xieyunwei/occupancy_networks/external/Choy2016/ShapeNetRendering/'
PRE_BUILD_ROOT = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'
#DATASET_PATH = '/home1/xieyunwei/ShapeNet.build.direct/'
DATASET_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'

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
            rendering_curr_model_root = os.path.join(DATASET_PATH, model_class, '4_points')
            if not os.path.exists(os.path.join(rendering_curr_model_root, '%s.npz' % model_id)):
                #print('%s/%s is missing' % (model_class, model_id))
                missing_count += 1
                '''
                in_path = os.path.join(PRE_BUILD_ROOT, model_class, '2_watertight', '%s.off' % model_id)
                mesh = trimesh.load(in_path, process=False)
                bbox = mesh.bounding_box.bounds

                # Compute location and scale
                loc = (bbox[0] + bbox[1]) / 2
                scale = (bbox[1] - bbox[0]).max() / (1)
                mesh.apply_translation(-loc)
                mesh.apply_scale(1 / scale)

                ratio_world = mesh.volume / ((1+0.1) ** 3)
                ratio_bbox = mesh.volume / mesh.bounding_box.volume
                print('Volume ratio, world:%.4f; bbox:%.4f' % (ratio_world, ratio_bbox) )
                '''
                continue
            
            #pack = np.load(os.path.join(rendering_curr_model_root, '%s.npz' % model_id))
            #if pack['points'].shape[0] != 50000:
            #    missing_count += 1
            #    continue

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

def fix_build():
    print('Counting:')
    for model_class in CLASSES:
        missing_count = 0

        class_root = os.path.join(SHAPENET_ROOT, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_root)

        class_points_root = os.path.join(DATASET_PATH, model_class, '4_points_direct')
        class_pointcloud_root = os.path.join(DATASET_PATH, model_class, '4_pointcloud_direct')
        for model_id in tqdm(current_class_ids):
            model_points_file = os.path.join(class_points_root, '%s.npz' % model_id)
            model_pointcloud_file = os.path.join(class_pointcloud_root, '%s.npz' % model_id)
            if os.path.exists(os.path.join(class_points_root, '%s.npz' % model_id)):
                #print('%s/%s fixed' % (model_class, model_id))
                point_dict = np.load(model_points_file)
                true_scale = point_dict['scale']
                true_loc = point_dict['loc']

                pointcloud_data = np.load(model_pointcloud_file)
                pointcloud = pointcloud_data['points']
                normals = pointcloud_data['normals']
                np.savez(model_pointcloud_file, points=pointcloud, 
                    normals=normals, loc=true_loc, scale=true_scale)


def check_bounds():
    for model_class in CLASSES:
        class_points_root = os.path.join(DATASET_PATH, model_class, '0_in')
        current_file_names = np.array(os.listdir(class_points_root))

        idx = np.random.choice(len(current_file_names), 10, replace=False)
        
        info = []
        current_file_names = current_file_names[idx]
        for filename in tqdm(current_file_names):
            mesh_file = os.path.join(class_points_root, filename)
            mesh = trimesh.load(mesh_file)
            bbox = mesh.bounding_box.bounds
            scale = (bbox[1] - bbox[0]).max()

            info.append(scale)

        print('CLASS: %s, max: %f, min: %f, avg: %f' % (model_class, max(info), min(info), sum(info)/len(info) ))
                

if __name__ == '__main__':
    #main()
    #fix_build()
    
    check_bounds()

