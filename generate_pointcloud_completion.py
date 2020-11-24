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

# Arguments
parser = argparse.ArgumentParser(
    description='Generate pointcloud completion results.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--out_folder_name', type=str, default='depth_pointcloud_completion.direct', 
                    help='Output folder name (under original dataset folder)')
parser.add_argument('--batch_size', type=int, default=256, help='Generation batch size.')
parser.add_argument('--combine_pc', action='store_true', help='Combine input and predict.')
args = parser.parse_args()

### rename ###
out_folder_name = args.out_folder_name

cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

print('input_type:', cfg['data']['input_type'])
assert cfg['data']['input_type'] == 'depth_pointcloud'
depth_pc_dir = cfg['data']['depth_pointcloud_root']
out_dir = depth_pc_dir

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
train_dataset = data.Shapes3dDataset_AllImgs(dataset_folder, fields, split=None)

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

it = 0
for batch in tqdm(train_loader):
    it += 1
    model.eval()

    encoder_inputs, raw_data = compose_inputs(batch, mode='train', device=device, input_type='depth_pointcloud',
                                                depth_pointcloud_transfer=depth_pointcloud_transfer)
    cur_batch_size = encoder_inputs.size(0)
    idxs = batch.get('idx')
    viewids = batch.get('viewid')

    kwargs = {}
    if model.space_carver_mode:
        kwargs = organize_space_carver_kwargs(
            model.space_carver_mode, kwargs, 
            raw_data, batch, device
        )

    with torch.no_grad():
        pointcloud_hat = model(encoder_inputs, **kwargs)

        if isinstance(pointcloud_hat, tuple):
            pointcloud_hat,_ = pointcloud_hat

        if args.combine_pc:
            pointcloud_hat = torch.cat([pointcloud_hat, encoder_inputs], dim=1)
            pointcloud_hat_flipped = pointcloud_hat.transpose(1, 2).contiguous()
            pointcloud_hat_idx = pointnet2_utils.furthest_point_sample(pointcloud_hat_flipped, 2048)
            pointcloud_hat = pointnet2_utils.gather_operation(pointcloud_hat_flipped, pointcloud_hat_idx).transpose(1, 2).contiguous()
    
    for i in range(cur_batch_size):
        cur_pointcloud_hat = pointcloud_hat[i].cpu().numpy()

        cur_model_info = train_dataset.get_model_dict(idxs[i]) # category & model
        cur_viewid = viewids[i]

        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], out_folder_name)):
            os.mkdir(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], out_folder_name))
        
        save_pc_path = os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'],
            out_folder_name, '%.2d_pointcloud.npz' % cur_viewid)

        np.savez(save_pc_path, pointcloud=cur_pointcloud_hat)    
