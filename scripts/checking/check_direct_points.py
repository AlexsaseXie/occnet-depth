import os
import sys
import time
import datetime
import trimesh
import numpy as np
from tqdm import tqdm
import trimesh
import shutil

SHAPENET_ROOT = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/'
R2N2_ROOT = '/home2/xieyunwei/occupancy_networks/external/Choy2016/ShapeNetRendering/'
PRE_BUILD_ROOT = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'
DATASET_PATH = '/home1/xieyunwei/ShapeNet.build.direct/'
CHECK_OUTPUT_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build.direct.check/'

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

def pcwrite(filename, xyzrgb, nxnynz=None, color=True, normal=False):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    if color:
        rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    if color:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
    if normal:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
        ))

        if color:
            ply_file.write(" %d %d %d"%(
                rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
        
        if normal:
            ply_file.write(" %f %f %f" % (
                nxnynz[i, 0], nxnynz[i, 1], nxnynz[i, 2]
            ))
        
        ply_file.write("\n")

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a)

def main():
    mkdir_p(CHECK_OUTPUT_PATH)
    for model_class in CLASSES:
        class_points_root = os.path.join(DATASET_PATH, model_class, '4_points_direct')
        class_pointcloud_root = os.path.join(DATASET_PATH, model_class, '4_pointcloud_direct')
        current_class_ids = os.listdir(class_pointcloud_root)
        current_class_ids = list(map(lambda x: x.split('.')[0], current_class_ids))

        current_class_ids = np.array(current_class_ids)
        idx = np.random.choice(len(current_class_ids), 10, replace = False)
        current_class_ids = current_class_ids[idx]

        check_output_class_root = os.path.join(CHECK_OUTPUT_PATH, model_class)
        mkdir_p(check_output_class_root)
        for model_id in tqdm(current_class_ids):
            model_points_file = os.path.join(class_points_root, '%s.npz' % model_id)
            model_pointcloud_file = os.path.join(class_pointcloud_root, '%s.npz' % model_id)
            if os.path.exists(os.path.join(class_points_root, '%s.npz' % model_id)):
                check_output_model_root = os.path.join(check_output_class_root, model_id)
                mkdir_p(check_output_model_root)

                point_dict = np.load(model_points_file)
                points = point_dict['points']
                occupancies = np.unpackbits(point_dict['occupancies'])
                true_scale = point_dict['scale']
                true_loc = point_dict['loc']

                pointcloud_data = np.load(model_pointcloud_file)
                pointcloud = pointcloud_data['points']
                normals = pointcloud_data['normals']
                
                points_model = points * true_scale + true_loc
                # output occ
                inside_points = points_model[occupancies == 1]
                ply_path = os.path.join(check_output_model_root, 'inside_occ.ply')
                pcwrite(ply_path, inside_points, color=False)

                outside_points = points_model[occupancies == 0]
                ply_path = os.path.join(check_output_model_root, 'outside_occ.ply')
                pcwrite(ply_path, outside_points, color=False)

                # output pointcloud
                pointcloud_model = pointcloud * true_scale + true_loc
                ply_path = os.path.join(check_output_model_root, 'pointcloud.ply')
                pcwrite(ply_path, pointcloud_model, nxnynz=normals, color=False, normal=True)

                # copy raw model
                raw_in_model_file = os.path.join(PRE_BUILD_ROOT, model_class, '0_in', '%s.off' % model_id)
                off_path = os.path.join(check_output_model_root, 'in.off')
                shutil.copy(raw_in_model_file, off_path)

                

if __name__ == '__main__':
    main()