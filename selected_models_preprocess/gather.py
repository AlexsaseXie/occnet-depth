import os
import argparse
import shutil
import pickle
from tqdm import tqdm
import numpy as np

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

parser = argparse.ArgumentParser(
    description='Select Model'
)
parser.add_argument('--out_dir', type=str, default='./selected_models_preprocess_1/', help='Out dir') 
parser.add_argument('--out_list', type=str, default='./selected_models_preprocess_1/list.txt', help='Out txt')
args = parser.parse_args()

CLASSES = ['02691156', 
'02958343', 
'03001627', 
'03636649', 
'04256520', 
'04401088']

SAL_root = '/home2/xieyunwei/occupancy_networks/out/sal/z0/all_class_training_schedule/%s/%d/generation_150000/'
Gt_root = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'
New_root = '/home1/xieyunwei/ShapeNet.build.direct/'

def mkdir_p(root):
    if not os.path.exists(root):
        os.mkdir(root) 

def copy_p(a, b):
    if os.path.exists(a):
        shutil.copyfile(a, b)

def gather(class_id, it):
    SAL_path = SAL_root % (class_id, it)

    class_root = os.path.join(args.out_dir, class_id)
    mkdir_p(class_root)

    modelnames = os.listdir(SAL_path)
    for info in modelnames:
        txt = info.split('.')[0]
        if len(txt) > 5:
            modelname = txt
            break

    model_root = os.path.join(class_root, modelname)
    mkdir_p(model_root)

    # gt
    if os.path.exists(Gt_root):
        target_file = os.path.join(Gt_root, class_id, "0_in", "%s.off" % modelname)
        output_file = os.path.join(model_root, "gt.off")
        copy_p(target_file, output_file)

        target_file = os.path.join(Gt_root, class_id, "2_watertight", "%s.off" % modelname)
        output_file = os.path.join(model_root, "onet_watertight.off")
        copy_p(target_file, output_file)

        target_file = os.path.join(Gt_root, class_id, "4_points", "%s.npz" % modelname)
        output_file = os.path.join(model_root, "onet_points.npz")
        copy_p(target_file, output_file)

        data = np.load(target_file)
        points = data['points']
        occ = np.unpackbits(data['occupancies'])
        points = points[occ > 0]
        output_file = os.path.join(model_root, "onet_inside_points.ply")
        pcwrite(output_file, points, color=False)

        target_file = os.path.join(Gt_root, class_id, "4_pointcloud", "%s.npz" % modelname)
        output_file = os.path.join(model_root, "onet_pointcloud.npz")
        copy_p(target_file, output_file)

        data = np.load(target_file)
        points = data['points']
        #normals = data['normals']
        output_file = os.path.join(model_root, "onet_pointcloud.ply")
        pcwrite(output_file, points, color=False)



    if os.path.exists(New_root):
        target_file = os.path.join(New_root, class_id, "4_points_direct_tsdf0.008", "%s.npz" % modelname)
        output_file = os.path.join(model_root, "new_points.npz")
        copy_p(target_file, output_file)

        data = np.load(target_file)
        points = data['points']
        tsdf = data['tsdf']
        points = points[tsdf < 0]
        output_file = os.path.join(model_root, "new_inside_points.ply")
        pcwrite(output_file, points, color=False)

        target_file = os.path.join(New_root, class_id, "4_pointcloud_direct", "%s.npz" % modelname)
        output_file = os.path.join(model_root, "new_pointcloud.npz")
        copy_p(target_file, output_file)

        data = np.load(target_file)
        points = data['points']
        #normals = data['normals']
        output_file = os.path.join(model_root, "new_pointcloud.ply")
        pcwrite(output_file, points, color=False)


    return class_id + '\t' + modelname

mkdir_p(args.out_dir)

info_list = []
print('Start gathering')
for c in tqdm(CLASSES):
    for it in range(1):
        info = gather(c, it)
        info_list.append(info)

with open(args.out_list, 'w') as f:
    f.write('\n'.join(info_list))

