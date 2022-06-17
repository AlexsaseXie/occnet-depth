import os
import argparse
import shutil
import pickle

parser = argparse.ArgumentParser(
    description='Select Model'
)
parser.add_argument('--out_dir', type=str, default='selected_models', help='Out dir') 
parser.add_argument('--selected_dataset', type=str, default='selected_models/selected_best_full.pkl', help='Dataset')
args = parser.parse_args()

DISN_root = '/home2/xieyunwei/DISN/checkpoint/224_retrain/direct_pad_test_objs/65_0.0/'
IMNet_root = '/home2/xieyunwei/IM-NET-pytorch/samples/im_svr_224_all_out/'
ONet_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/onet/generation_simplify/meshes/'
ONet_depth_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/phase2_depth_pointcloud_MSN_4096_pointnet2_4layers_version2(dropout)_local_clean/generation_space_carved/meshes/'
Atlasnet_root = '/home2/xieyunwei/AtlasNet/svr_224_reconstructed/'
Pix2Mesh_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/pixel2mesh/generation/meshes/'
Gt_root = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'
Input_root = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.with_depth.10w10w/'

def mkdir_p(root):
    if not os.path.exists(root):
        os.mkdir(root) 

def copy_p(a, b):
    if os.path.exists(a):
        shutil.copyfile(a, b)

def gather(class_id, modelname):
    class_root = os.path.join(args.out_dir, class_id)
    mkdir_p(class_root)
    model_root = os.path.join(class_root, modelname)
    mkdir_p(model_root)
    # # onet
    # if os.path.exists(ONet_root):
    #     target_file = os.path.join(ONet_root, class_id, "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "onet.off")
    #     copy_p(target_file, output_file)

    # # onet_depth
    # if os.path.exists(ONet_depth_root):
    #     target_file = os.path.join(ONet_depth_root, class_id, "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "ours.off")
    #     copy_p(target_file, output_file)
    # # disn
    # if os.path.exists(DISN_root):
    #     target_file = os.path.join(DISN_root, class_id, "%s_%s_00_normalized.obj" % (class_id, modelname))
    #     output_file = os.path.join(model_root, "disn.obj")
    #     copy_p(target_file, output_file)
    # # imnet
    # if os.path.exists(IMNet_root):
    #     target_file = os.path.join(IMNet_root, class_id, modelname, 'vox_normalized.ply')
    #     output_file = os.path.join(model_root, "imnet.ply")
    #     copy_p(target_file, output_file)
    # # atlasnet
    # if os.path.exists(Atlasnet_root):
    #     target_file = os.path.join(Atlasnet_root, class_id, '%s_normalized.ply' % modelname)
    #     output_file = os.path.join(model_root, "atlasnet.ply")
    #     copy_p(target_file, output_file)

    # # pix2mesh
    # if os.path.exists(Pix2Mesh_root):
    #     target_file = os.path.join(Pix2Mesh_root, class_id, "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "pix2mesh.off")
    #     copy_p(target_file, output_file)

    # # gt
    # if os.path.exists(Gt_root):
    #     target_file = os.path.join(Gt_root, class_id, "0_in", "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "gt.off")
    #     copy_p(target_file, output_file)

    #     target_file = os.path.join(Gt_root, class_id, "2_watertight", "%s.off" % modelname)
    #     output_file = os.path.join(model_root, "gt_watertight.off")
    #     copy_p(target_file, output_file)

    # input
    if os.path.exists(Input_root):
        # target_file = os.path.join(Input_root, class_id, modelname, "depth", "00_depth.png")
        # output_file = os.path.join(model_root, "gt_depth.png")
        # copy_p(target_file, output_file)

        # target_file = os.path.join(Input_root, class_id, modelname, "img", "00_rgb.png")
        # output_file = os.path.join(model_root, "gt_rgb.png")
        # copy_p(target_file, output_file)

        # target_file = os.path.join(Input_root, class_id, modelname, "mask", "00_mask.png")
        # output_file = os.path.join(model_root, "gt_mask.png")
        # copy_p(target_file, output_file)

        target_file = os.path.join(Input_root, class_id, modelname, "img", "cameras.npz")
        output_file = os.path.join(model_root, "cameras.npz")
        copy_p(target_file, output_file)

with open(args.selected_dataset, 'rb') as f:
    m = pickle.load(f)

mkdir_p(args.out_dir)

from tqdm import tqdm
pbar = tqdm(total=m.shape[0])
for index, row in m.iterrows():
    class_id = row['class id']
    modelname = row['modelname']

    gather(class_id, modelname)

    pbar.update(1)
pbar.close()
