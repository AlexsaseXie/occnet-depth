import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import sys
import argparse
import time
sys.path.append('./')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
from im2mesh.point_completion.training import compose_inputs, compose_pointcloud, organize_space_carver_kwargs
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from tqdm import tqdm
from scripts.pix3d_preprocess import utils as pix3d_utils

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

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if 'depth_pointcloud_transfer' in cfg['model']:
    depth_pointcloud_transfer = cfg['model']['depth_pointcloud_transfer']
else:
    depth_pointcloud_transfer = 'world_scale_model'

dataset_folder = cfg['data']['path']

# Dataset
train_dataset = config.get_dataset('test', cfg, return_idx=True)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
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
for batch in tqdm(train_loader):
    it += 1
    model.eval()

    encoder_inputs, raw_data = compose_inputs(batch, mode='train', device=device, input_type='depth_pointcloud',
                                                depth_pointcloud_transfer=depth_pointcloud_transfer)
    cur_batch_size = encoder_inputs.size(0)
    idxs = batch.get('idx')

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

        idx = idxs[i]
        cur_image_info = train_dataset.get_info(idx) # image_info
        cur_image_category = cur_image_info['category']
        cur_image_name = pix3d_utils.get_image_name(cur_image_info)

        if not os.path.exists(os.path.join(out_dir, cur_image_category)):
            os.mkdir(os.path.join(out_dir, cur_image_category))
        
        if not os.path.exists(os.path.join(out_dir, cur_image_category, cur_image_name)):
            os.mkdir(os.path.join(out_dir, cur_image_category, cur_image_name))

        if not os.path.exists(os.path.join(out_dir, cur_image_category, cur_image_name, out_folder_name)):
            os.mkdir(os.path.join(out_dir, cur_image_category, cur_image_name, out_folder_name))
        
        save_pc_path = os.path.join(out_dir, cur_image_category, cur_image_name,
            out_folder_name, '00_pointcloud.npz')

        np.savez(save_pc_path, pointcloud=cur_pointcloud_hat)    
