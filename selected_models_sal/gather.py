import os
import argparse
import shutil
import pickle
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(
    description='Select Model'
)
parser.add_argument('--out_dir', type=str, default='./selected_models_sal/', help='Out dir') 
parser.add_argument('--out_list', type=str, default='./selected_models_sal/list.txt', help='Out txt')
args = parser.parse_args()

CLASSES = ['02691156', 
'02958343', 
'03001627', 
'03636649', 
'04256520', 
'04401088']

SPSR_root = '/home3/xieyunwei/ShapeNet.SAL.clean/%s/%s/spsr_fps30000/00_recon.ply'
SAL_root = '/home2/xieyunwei/occupancy_networks/out/sal/z0/all_class_training_schedule/%s/%d/generation_150000/'
SAIL_S3_root = '/home2/xieyunwei/occupancy_networks/out/sail_s3_z_fixed/kmeans/voxelize_schedule/%s_K40_null/%d/generation_mean_simple/'
Gt_root = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'
Gt_SAL_root = '/home3/xieyunwei/ShapeNet.build.direct_remove'

def mkdir_p(root):
    if not os.path.exists(root):
        os.mkdir(root) 

def copy_p(a, b):
    if os.path.exists(a):
        shutil.copyfile(a, b)

def gather(class_id, it):
    SAL_path = SAL_root % (class_id, it)
    SAIL_S3_path = SAIL_S3_root % (class_id, it)

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

    # # sal
    # if os.path.exists(SAL_path):
    #     target_file = os.path.join(SAL_path, "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "sal.off")
    #     copy_p(target_file, output_file)

    # # sail-s3
    # if os.path.exists(SAL_path):
    #     target_file = os.path.join(SAIL_S3_path, "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "sail_s3.off")
    #     copy_p(target_file, output_file)

    # #spsr
    # SPSR_file = SPSR_root % (class_id, modelname)
    # if os.path.exists(SPSR_file):
    #     output_file = os.path.join(model_root, "spsr.ply")
    #     copy_p(SPSR_file, output_file)

    # # gt
    # if os.path.exists(Gt_root):
    #     target_file = os.path.join(Gt_root, class_id, "0_in", "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "gt.off")
    #     copy_p(target_file, output_file)

    #     target_file = os.path.join(Gt_root, class_id, "2_watertight", "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "gt_watertight.off")
    #     copy_p(target_file, output_file)

    # gt point cloud
    if os.path.exists(Gt_SAL_root):
        target_file = os.path.join(Gt_SAL_root, class_id, "4_pointcloud_direct_fps_N30000_copy", "%s.npz" % modelname)
        output_file = os.path.join(model_root, "input_pc_30000.npz")
        copy_p(target_file, output_file)

    return class_id + '\t' + modelname

mkdir_p(args.out_dir)

info_list = []
print('Start gathering')
for c in tqdm(CLASSES):
    for it in range(5):
        info = gather(c, it)
        info_list.append(info)

with open(args.out_list, 'w') as f:
    f.write('\n'.join(info_list))

