import argparse
# import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
import sys

sys.path.append('./')
from im2mesh import config, data
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.io import load_pointcloud
from torchvision import transforms
from im2mesh.data.transforms import PointcloudOffset
from scripts.pix3d_preprocess import utils as pix3d_utils


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--eval_input', action='store_true',
                    help='Evaluate inputs instead.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
if not args.eval_input:
    out_file = os.path.join(generation_dir, 'eval_meshes_new_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_meshes_new.csv')
else:
    out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
    out_file_class = os.path.join(generation_dir, 'eval_input.csv')

# Dataset
if 'test_range' in cfg['data']:
    input_range = cfg['data']['test_range']
    print('Test range:', input_range)
else:
    input_range = None

by_tsdf = None
if 'test_tsdf' in cfg['data']:
    by_tsdf = cfg['data']['test_tsdf']
    print('test by tsdf:', by_tsdf)

points_field = data.Pix3d_PointField(
    build_path=cfg['data']['pix3d_build_path'],
    unpackbits=cfg['data']['points_unpackbits'],
    input_range=input_range,
    by_tsdf=by_tsdf
)

if by_tsdf is not None:
    pc_transform = transforms.Compose([
        PointcloudOffset(by_tsdf)
    ])
    print('Point cloud offset:', by_tsdf)
else:
    pc_transform = None
 
pointcloud_field = data.Pix3d_PointCloudField(
    build_path=cfg['data']['pix3d_build_path'],
    transform=pc_transform
)

fields = {
    'points_iou': points_field,
    'pointcloud_chamfer': pointcloud_field,
    'idx': data.IndexField(),
}

if 'pix3d_root' in cfg['data']:
    pix3d_root = cfg['data']['pix3d_root']
else:
    pix3d_root = '.'

dataset_folder = cfg['data']['path']
dataset = data.Pix3dDataset(dataset_folder, fields, pix3d_root=pix3d_root)

# Evaluator
evaluator = MeshEvaluator(n_points=100000)

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Evaluate all classes
eval_dicts = []
print('Evaluating meshes...')
for it, data in enumerate(tqdm(test_loader)):
    if data is None:
        print('Invalid data.')
        continue

    # Output folders
    if not args.eval_input:
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
    else:
        mesh_dir = os.path.join(generation_dir, 'input')
        pointcloud_dir = os.path.join(generation_dir, 'input')

    # Get index etc.
    idx = data['idx'].item()

    cur_image_info = dataset.get_info(idx) # image_info
    cur_image_category = cur_image_info['category']
    cur_image_name = pix3d_utils.get_image_name(cur_image_info)
    cur_image_model_name = pix3d_utils.get_model_name(cur_image_info)


    mesh_dir = os.path.join(mesh_dir, cur_image_category)
    pointcloud_dir = os.path.join(pointcloud_dir, cur_image_category)

    # Evaluate
    try:
        pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
        normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
        points_tgt = data['points_iou'].squeeze(0).numpy()
        occ_tgt = data['points_iou.occ'].squeeze(0).numpy()
    except:
        continue
    finally:
        pass

    # Evaluating mesh and pointcloud
    # Start row and put basic informatin inside
    eval_dict = {
        'idx': idx,
        'class id': cur_image_category,
        'image name': cur_image_name,
        'model name': cur_image_model_name,
    }
    eval_dicts.append(eval_dict)

    # Evaluate mesh
    if cfg['test']['eval_mesh']:
        mesh_file = os.path.join(mesh_dir, '%s.off' % cur_image_name)

        if os.path.exists(mesh_file):
            mesh = trimesh.load(mesh_file, process=False)
            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    # Evaluate point cloud
    if cfg['test']['eval_pointcloud']:
        pointcloud_file = os.path.join(
            pointcloud_dir, '%s.ply' % cur_image_name)

        if os.path.exists(pointcloud_file):
            pointcloud = load_pointcloud(pointcloud_file)
            eval_dict_pcl = evaluator.eval_pointcloud(
                pointcloud, pointcloud_tgt)
            for k, v in eval_dict_pcl.items():
                eval_dict[k + ' (pcl)'] = v
        else:
            print('Warning: pointcloud does not exist: %s'
                    % pointcloud_file)

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file  with main statistics
eval_df_class = eval_df.groupby(by=['class id']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
