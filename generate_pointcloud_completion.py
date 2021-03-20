import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import sys
import argparse
import time
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
from im2mesh.point_completion.training import compose_inputs, compose_pointcloud, organize_space_carver_kwargs
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from tqdm import tqdm
import pandas as pd

# Arguments
parser = argparse.ArgumentParser(
    description='Generate pointcloud completion results.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--out_dir', type=str, default='default',
                    help='Output dir name (default is the same as depth pointcloud folder)')
parser.add_argument('--out_folder_name', type=str, default='depth_pointcloud_completion.direct', 
                    help='Output folder name (under original dataset folder)')
parser.add_argument('--batch_size', type=int, default=256, help='Generation batch size.')
parser.add_argument('--combine_pc', action='store_true', help='Combine input and predict.')
parser.add_argument('--resample', type=int, default=0, help='random resample points count.')
parser.add_argument('--time_test', action='store_true', help='time test')
args = parser.parse_args()

### rename ###
out_folder_name = args.out_folder_name

cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

print('input_type:', cfg['data']['input_type'])
assert cfg['data']['input_type'] == 'depth_pointcloud'
depth_pc_dir = cfg['data']['depth_pointcloud_root']
if args.out_dir == 'default':
    out_dir = depth_pc_dir
else:
    out_dir = args.out_dir

if not os.path.exists(out_dir) and not args.time_test:
    os.mkdir(out_dir)

if 'depth_pointcloud_transfer' in cfg['model']:
    depth_pointcloud_transfer = cfg['model']['depth_pointcloud_transfer']
else:
    depth_pointcloud_transfer = 'world_scale_model'

dataset_folder = cfg['data']['path']

# Dataset
def get_fields():
    fields = {}
    input_field = config.get_inputs_field('train', cfg)
    fields['inputs'] = input_field
    fields['idx'] = data.IndexField()
    fields['viewid'] = data.ViewIdField()
    return fields

fields = get_fields()
if not args.time_test:
    train_dataset = data.Shapes3dDataset_AllImgs(dataset_folder, fields, split=None)
else:
    train_dataset = data.Shapes3dDataset_AllImgs(dataset_folder, fields, split='test', n_views=1)

batch_size = args.batch_size if not args.time_test else 1
n_workers = 4 if not args.time_test else 0
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

checkpoint_io = CheckpointIO(cfg['training']['out_dir'], model=model)
try:
    load_dict = checkpoint_io.load('model_best.pt', strict=True)
except FileExistsError:
    load_dict = dict()

method = cfg['method']
if method == 'point_completion':
    method = 'FCAE'

it = 0
time_dicts = []
for batch in tqdm(train_loader):
    it += 1
    model.eval()

    encoder_inputs, raw_data = compose_inputs(batch, mode='train', device=device, input_type='depth_pointcloud',
                                                depth_pointcloud_transfer=depth_pointcloud_transfer)
    cur_batch_size = encoder_inputs.size(0)
    idxs = batch.get('idx')
    viewids = batch.get('viewid')

    kwargs = {}
    if getattr(model, 'module', False):
        space_carver_mode = getattr(model.module, 'space_carver_mode', False)
    else:
        space_carver_mode = getattr(model, 'space_carver_mode', False)
    if space_carver_mode:
        target_space = getattr(cfg['model'], 'gt_pointcloud_transfer', 'world_scale_model')
        kwargs = organize_space_carver_kwargs(
            space_carver_mode, kwargs, 
            raw_data, batch, device,
            target_space=target_space
        )

    if args.time_test:
        single_t0 = time.time()

    with torch.no_grad():
        pointcloud_hat = model(encoder_inputs, **kwargs)

        if method == 'FCAE':
            if isinstance(pointcloud_hat, tuple):
                pointcloud_hat,_ = pointcloud_hat
        elif method == 'MSN': 
            if isinstance(pointcloud_hat, tuple):
                _, pointcloud_hat, _ = pointcloud_hat
        else:
            raise NotImplementedError

        if args.combine_pc:
            pointcloud_hat = torch.cat([pointcloud_hat, encoder_inputs], dim=1)
            pointcloud_hat_flipped = pointcloud_hat.transpose(1, 2).contiguous()
            pointcloud_hat_idx = pointnet2_utils.furthest_point_sample(pointcloud_hat_flipped, 2048)
            pointcloud_hat = pointnet2_utils.gather_operation(pointcloud_hat_flipped, pointcloud_hat_idx).transpose(1, 2).contiguous()
    
    for i in range(cur_batch_size):
        cur_pointcloud_hat = pointcloud_hat[i].cpu().numpy()

        if args.resample != 0:
            cur_pointcloud_hat = np.unique(cur_pointcloud_hat, axis=0)
            if cur_pointcloud_hat.shape[0] >= args.resample:
                idx = np.random.choice(cur_pointcloud_hat.shape[0], size=args.resample, replace=False)
            else:
                idx = np.random.randint(cur_pointcloud_hat.shape[0], size=args.resample)
            cur_pointcloud_hat = cur_pointcloud_hat[idx, :] 

        cur_model_info = train_dataset.get_model_dict(idxs[i]) # category & model
        cur_viewid = viewids[i]

        if args.time_test:
            time_dict = {
                'idx': it,
                'class id': cur_model_info['category'],
                'modelname': cur_model_info['model'],
                'viewid': cur_viewid,
                'time (depth estimation)': time.time() - single_t0
            }

            time_dicts.append(time_dict)
            break

        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'])):
            os.mkdir(os.path.join(out_dir, cur_model_info['category']))
        
        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'])):
            os.mkdir(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model']))

        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], out_folder_name)):
            os.mkdir(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], out_folder_name))
        
        save_pc_path = os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'],
            out_folder_name, '%.2d_pointcloud.npz' % cur_viewid)

        np.savez(save_pc_path, pointcloud=cur_pointcloud_hat)    

if args.time_test:
    if not os.path.exists('./timings/'):
        os.mkdir('./timings/')

    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle('./timings/point_cloud_completion_time_full.pkl')

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class id']).mean()
    time_df_class.to_pickle('./timings/point_cloud_completion_time_cls.pkl')

    time_df_class.loc['mean'] = time_df_class.mean()
    print('Timings: ', time_df_class)
