import argparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
from im2mesh import data as im_data
from im2mesh.eval import MeshEvaluator, EMPTY_PCL_DICT, EMPTY_PCL_DICT_NORMALS
from im2mesh.utils.io import load_pointcloud
from im2mesh.utils.libmesh import check_mesh_contains
import time

parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('disn_checkpoint', type=str, help='Path to config file.')
parser.add_argument('--disn_gt', type=str, default='/home2/xieyunwei/DISN_data/marching_cubes', 
    help='Path to disn marching cube gt surface')
parser.add_argument('--pymesh', action='store_true', help='Use pymesh voxelize')
parser.add_argument('--compare_disn', action='store_true', help='Compare with disn marching cube gt')

args = parser.parse_args()

# Shorthands
out_dir = args.disn_checkpoint
generation_dir = out_dir

if args.pymesh:
    import pymesh
    print('Using pymesh')
    save_f = "pymesh_%s"
else:
    save_f = "%s"

if not args.compare_disn:
    out_file = os.path.join(generation_dir, save_f % 'eval_meshes_full.pkl')
    out_file_class = os.path.join(generation_dir, save_f % 'eval_meshes.csv')
else:
    out_file = os.path.join(generation_dir, save_f % 'eval_meshes_disn_gt_full.pkl')
    out_file_class = os.path.join(generation_dir, save_f % 'eval_meshes_disn_gt.csv')

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

    print('Test split: ', 'disn_test')

    dataset_folder = './data/ShapeNet/'
    dataset = im_data.Shapes3dDataset(
        dataset_folder, fields,
        'disn_test',
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

        # Mesh folders
        mesh_dir = generation_dir

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

        assert category_id != 'n/a'
        mesh_dir = os.path.join(mesh_dir, category_id)

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
        mesh_file = os.path.join(mesh_dir, '%s_%s_00.obj' % (category_id,modelname))
        gt_marching_cube_file = os.path.join(args.disn_gt, category_id, modelname, 'isosurf.obj')

        if os.path.exists(mesh_file):
            gt_mesh = trimesh.load(gt_marching_cube_file, process=False)
            # transform
            bbox = gt_mesh.bounding_box.bounds

            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()

            # Transform input mesh
            if not args.pymesh:
                mesh = trimesh.load(mesh_file, process=False)
                mesh.apply_translation(-loc)
                mesh.apply_scale(1 / scale)
            else:
                s_time = time.time()
                mesh = pymesh.load_mesh(mesh_file)
                mesh = pymesh.form_mesh((mesh.vertices - loc) * (1. / scale), mesh.faces, mesh.voxels)
                print('load & resize mesh:', time.time() - s_time)
                s_time = time.time()

                grid = pymesh.VoxelGrid( 1. / 128. )
                grid.insert_mesh(mesh) 
                grid.create_grid() 
                print('create grid:', time.time() - s_time)
                s_time = time.time()

                mesh = trimesh.Trimesh(vertices=grid.mesh.vertices, faces=grid.mesh.faces)
                print('create trimesh', time.time() - s_time)


            eval_dict_mesh = evaluator.eval_mesh(
                mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)
            
            if eval_dict_mesh['iou'] != -1:
                eval_dicts.append(eval_dict)
            else:
                print('skip model ', mesh_file)


            for k, v in eval_dict_mesh.items():
                eval_dict[k + ' (mesh)'] = v
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    return eval_dicts

def evaluate_disn_gt_mesh():
    points_field = im_data.PointsField(
        'points.npz', 
        unpackbits=True,
    )

    fields = {
        'points_iou': points_field,
        'idx': im_data.IndexField(),
    }

    print('Test split: ', 'disn_test')

    dataset_folder = './data/ShapeNet/'
    dataset = im_data.Shapes3dDataset(
        dataset_folder, fields,
        'disn_test',
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

        # Mesh folders
        mesh_dir = generation_dir

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

        assert category_id != 'n/a'
        mesh_dir = os.path.join(mesh_dir, category_id)

        # Evaluate
        points_tgt = data['points_iou'].squeeze(0).numpy()

        # Evaluating mesh and pointcloud
        # Start row and put basic informatin inside
        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }

        # Evaluate mesh
        mesh_file = os.path.join(mesh_dir, '%s_%s_00.obj' % (category_id,modelname))
        gt_marching_cube_file = os.path.join(args.disn_gt, category_id, modelname, 'isosurf.obj')

        if os.path.exists(mesh_file):
            gt_mesh = trimesh.load(gt_marching_cube_file, process=False)
            # transform
            bbox = gt_mesh.bounding_box.bounds

            # Compute location and scale
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()

            # Transform input mesh
            if not args.pymesh:
                mesh = trimesh.load(mesh_file, process=False)
                if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
                    mesh.apply_translation(-loc)
                    mesh.apply_scale(1 / scale)
            else:
                mesh = pymesh.load_mesh(mesh_file)
                if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
                    mesh = pymesh.form_mesh((mesh.vertices - loc) * (1. / scale), mesh.faces, mesh.voxels)
                    grid = pymesh.VoxelGrid( 1. / 128. )
                    grid.insert_mesh(mesh)
                    grid.create_grid() 
                    mesh = grid.mesh

            if len(gt_mesh.vertices) != 0 and len(gt_mesh.faces) != 0:
                gt_mesh.apply_translation(-loc)
                gt_mesh.apply_scale(1 / scale)

                # sample on gt_mesh
                pointcloud_tgt, idx = gt_mesh.sample(100000, return_index=True)
                pointcloud_tgt = pointcloud_tgt.astype(np.float32)
                normals_tgt = gt_mesh.face_normals[idx]
                occ_tgt = check_mesh_contains(gt_mesh, points_tgt)

                eval_dict_mesh = evaluator.eval_mesh(
                    mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt)

                if eval_dict_mesh['iou'] != -1:
                    eval_dicts.append(eval_dict)
                else:
                    print('skip model ', mesh_file)

                for k, v in eval_dict_mesh.items():
                    eval_dict[k + ' (mesh)'] = v
            else:
                # Empty ground truth mesh
                print('Warning: empty ground truth mesh: %s' % gt_marching_cube_file)
        else:
            print('Warning: mesh does not exist: %s' % mesh_file)

    return eval_dicts

if not args.compare_disn:
    eval_dicts = evaluate_occ_gt_mesh()
else:
    eval_dicts = evaluate_disn_gt_mesh()

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
