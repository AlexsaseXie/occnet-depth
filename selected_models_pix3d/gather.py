import os
import argparse
import shutil
import pickle

parser = argparse.ArgumentParser(
    description='Select Model'
)
parser.add_argument('--out_dir', type=str, default='selected_models_pix3d', help='Out dir') 
parser.add_argument('--selected_dataset', type=str, default='selected_models_pix3d/selected_best_full.pkl', help='Dataset')
args = parser.parse_args()

ONet_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/onet/generation_pix3d/meshes/'
ONet_depth_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/phase2_depth_pointcloud_MSN_4096_pointnet2_local/generation_pix3d/meshes/'
Gt_root = '/home2/xieyunwei/occupancy_networks/data/pix3d/pix3d.build/'
Input_root = '/home2/xieyunwei/occupancy_networks/data/pix3d/generation/'
    

def mkdir_p(root):
    if not os.path.exists(root):
        os.mkdir(root) 

def copy_p(a, b):
    if os.path.exists(a):
        shutil.copyfile(a, b)

def gather(class_id, imagename, modelname):
    class_root = os.path.join(args.out_dir, class_id)
    mkdir_p(class_root)
    model_root = os.path.join(class_root, imagename)
    mkdir_p(model_root)
    # onet
    if os.path.exists(ONet_root):
        target_file = os.path.join(ONet_root, class_id, "%s.off" % imagename)
        output_file = os.path.join(model_root, "onet.off")
        copy_p(target_file, output_file)

    # onet_depth
    if os.path.exists(ONet_depth_root):
        target_file = os.path.join(ONet_depth_root, class_id, "%s.off" % imagename)
        output_file = os.path.join(model_root, "ours.off")
        copy_p(target_file, output_file)
    
    # gt
    if os.path.exists(Gt_root):
        target_file = os.path.join(Gt_root, class_id, "0_in", "%s.off" % modelname)
        output_file = os.path.join(model_root, "gt.off")
        copy_p(target_file, output_file)

        target_file = os.path.join(Gt_root, class_id, "2_watertight", "%s.off" % modelname)
        output_file = os.path.join(model_root, "gt_watertight.off")
        copy_p(target_file, output_file)

    # input
    if os.path.exists(Input_root):
        target_file = os.path.join(Input_root, class_id, imagename, "%s_final.png" % imagename)
        output_file = os.path.join(model_root, "gt_rgb.png")
        copy_p(target_file, output_file)

        target_file = os.path.join(Input_root, class_id, imagename, "%s_final_mask.png" % imagename)
        output_file = os.path.join(model_root, "gt_mask.png")
        copy_p(target_file, output_file)


with open(args.selected_dataset, 'rb') as f:
    m = pickle.load(f)

mkdir_p(args.out_dir)

from tqdm import tqdm
pbar = tqdm(total=m.shape[0])
for index, row in m.iterrows():
    class_id = row['class id']
    imagename = row['image name']
    modelname = row['model name']

    gather(class_id, imagename, modelname)

    pbar.update(1)
pbar.close()
