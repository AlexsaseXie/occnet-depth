import argparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
import math
from im2mesh import data as im_data
from im2mesh.eval import MeshEvaluator, EMPTY_PCL_DICT, EMPTY_PCL_DICT_NORMALS
from im2mesh.utils.io import load_pointcloud
from im2mesh.utils.libmesh import check_mesh_contains
import time

parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
#parser.add_argument('atlasnet_dir', type=str, help='Path to imnet generation dir.')
parser.add_argument('--shapenet_gt', type=str, default='/home2/xieyunwei/occupancy_networks/data/ShapeNet.build', 
    help='Path to disn marching cube gt surface')

args = parser.parse_args()

# Shorthands
out_dir = '.'

out_file = os.path.join(out_dir, 'eval_meshes_full.pkl')
out_file_class = os.path.join(out_dir, 'eval_meshes.csv')

def evaluate_occ_gt_mesh():
    points_field = im_data.PointsField(
        'points.npz', 
        unpackbits=True,
    )
    
    pointcloud_field = im_data.PointCloudField(
        'pointcloud.npz'
    )

    fields = {
        'points_iou': points_field,
        'pointcloud_chamfer': pointcloud_field,
        'idx': im_data.IndexField(),
    }

    print('Test split: ', 'imnet_test')

    dataset_folder = './data/ShapeNet.with_depth.10w10w/'
    dataset = im_data.Shapes3dDataset(
        dataset_folder, fields,
        'imnet_test',
        categories=None
    )

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

        # Get index etc.
        idx = data['idx'].item()

        try:
            model_dict = dataset.get_model_dict(idx)
        except AttributeError:
            model_dict = {'model': str(idx), 'category': 'n/a'}
        
        modelname = model_dict['model']
        category_id = model_dict['category']

        try:
            category_name = dataset.metadata[category_id].get('name', 'n/a')
        except AttributeError:
            category_name = 'n/a'

        # Evaluate
        pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
        normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
        points_tgt = data['points_iou'].squeeze(0).numpy()
        occ_tgt = data['points_iou.occ'].squeeze(0).numpy()

        # Evaluating mesh and pointcloud
        # Start row and put basic informatin inside
        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }
        #eval_dicts.append(eval_dict)

        # Evaluate mesh
        #mesh_file = os.path.join(out_dir, category_id, '%s.ply' % modelname)
        gt_mesh_file = os.path.join(args.shapenet_gt, category_id, '0_in', '%s.off' % modelname)

        if os.path.exists(gt_mesh_file):
            raw_mesh = trimesh.load(gt_mesh_file, process=False)
            # transform
            bbox = raw_mesh.bounding_box.bounds

            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2.
            scale = (bbox[1] - bbox[0]).max() / 1.

            # Transform input mesh
            #if len(raw_mesh.vertices) >= 1 and len(raw_mesh.faces) >= 1:
            try:
                raw_mesh.apply_translation(-loc)
                raw_mesh.apply_scale(1 / scale)
                #raw_mesh.vertices = raw_mesh.vertices - loc
                #raw_mesh.vertices = raw_mesh.vertices * (1. / scale)
            except:
                continue
            finally:
                eval_dicts.append(eval_dict)

            eval_dict_mesh = evaluator.eval_mesh(
                raw_mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt, eval_iou=False)
            print('IoU:', eval_dict_mesh['iou'])
            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    return eval_dicts

eval_dicts = evaluate_occ_gt_mesh()

# Create pandas dataframe and save
eval_df = pd.DataFrame(eval_dicts)
eval_df.set_index(['idx'], inplace=True)
eval_df.to_pickle(out_file)

# Create CSV file with main statistics
eval_df_class = eval_df.groupby(by=['class name']).mean()
eval_df_class.to_csv(out_file_class)

# Print results
eval_df_class.loc['mean'] = eval_df_class.mean()
print(eval_df_class)
