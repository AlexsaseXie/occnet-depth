import torch
# import torch.distributions as dist
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import sys
sys.path.append('./')
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud
from im2mesh.utils.visualize import visualize_data
from im2mesh.utils.voxels import VoxelGrid


parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg['training']['out_dir']
out_time_file = os.path.join(out_dir, 'time_generation_full.pkl')
out_time_file_class = os.path.join(out_dir, 'time_generation.pkl')

input_type = cfg['data']['input_type']

# Dataset
dataset = config.get_dataset('train', cfg, return_idx=True)
# Model
model = config.get_model(cfg, device=device, dataset=dataset)
# Generator
generator = config.get_generator(model, cfg, device=device)
# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False
)

# Statistics
time_dicts = []
for it, data in enumerate(tqdm(test_loader)):
    model_out_dir = os.path.join(out_dir, '%d' % it)
    generation_dir = os.path.join(out_dir, '%d' % it, cfg['generation']['generation_dir'])

    checkpoint_io = CheckpointIO(model_out_dir, model=model)
    model_file = os.path.join(model_out_dir, cfg['test']['model_file'])
    if not os.path.exists(model_file):
        continue

    if not os.path.exists(generation_dir):
        os.mkdir(generation_dir)

    scalars = checkpoint_io.load(cfg['test']['model_file'])
    z_vec = scalars.get('z_vec', None)

    # Generate
    model.eval()

    # Get index etc.
    idx = data['idx'].item()

    try:
        model_dict = dataset.get_model_dict(idx)
    except AttributeError:
        model_dict = {'model': str(idx), 'category': 'n/a'}
    
    modelname = model_dict['model']
    category_id = model_dict.get('category', 'n/a')

    try:
        category_name = dataset.metadata[category_id].get('name', 'n/a')
    except AttributeError:
        category_name = 'n/a'
    
    # Timing dict
    time_dict = {
        'idx': idx,
        'class id': category_id,
        'class name': category_name,
        'modelname': modelname,
    }
    time_dicts.append(time_dict)

    t0 = time.time()
    out = generator.generate_mesh(data, z_prior=z_vec)
    time_dict['mesh'] = time.time() - t0

    # Get statistics
    if len(out) == 2:
        mesh, stats_dict = out
    elif len(out) == 3:
        mesh, refined_mesh, stats_dict = out
    else:
        raise NotImplementedError
    time_dict.update(stats_dict)

    # Write output
    mesh_out_file = os.path.join(generation_dir, '%s.off' % modelname)
    mesh.export(mesh_out_file)

# Create pandas dataframe and save
time_df = pd.DataFrame(time_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_pickle(out_time_file)

# Create pickle files  with main statistics
time_df_class = time_df.groupby(by=['class name']).mean()
time_df_class.to_pickle(out_time_file_class)

# Print results
time_df_class.loc['mean'] = time_df_class.mean()
print('Timings [s]:')
print(time_df_class)
