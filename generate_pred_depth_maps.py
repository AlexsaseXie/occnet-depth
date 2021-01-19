import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import sys
import argparse
import time
import matplotlib; matplotlib.use('Agg')
from im2mesh import config, data
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.visualize import visualize_data

# Arguments
parser = argparse.ArgumentParser(
    description='Generate depth predictions.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--out_dir', type=str, default='./data/ShapeNet.depth_pred.uresnet/', help='Output dir')
parser.add_argument('--out_folder_name', type=str, default='depth_pred', help='output folder name')
parser.add_argument('--batch_size', type=int, default=128) # 60 for hourglass
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = args.out_dir
print('out_dir:',out_dir)
pred_path = args.out_folder_name
batch_size = args.batch_size
print('input_type:', cfg['data']['input_type'])

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# Dataset
def get_fields():
    fields = {}
    input_field = config.get_inputs_field('train', cfg)
    fields['inputs'] = input_field
    fields['idx'] = data.IndexField()
    fields['viewid'] = data.ViewIdField()
    return fields

fields = get_fields()
train_dataset = data.Shapes3dDataset_AllImgs(cfg['data']['path'], fields, split=None)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=False,
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
batch_count = len(train_loader)
t0 = time.time()

if 'absolute_depth' in cfg['data']:
    absolute_depth = cfg['data']['absolute_depth']
else:
    absolute_depth = True

print('absolute_depth:', absolute_depth)

from tqdm import tqdm
pbar = tqdm(total=batch_count)
for batch in train_loader:
    it += 1
    model.eval()

    inputs = batch.get('inputs').to(device)
    cur_batch_size = inputs.size(0)
    gt_masks = batch.get('inputs.mask').to(device).byte()

    idxs = batch.get('idx')
    viewids = batch.get('viewid')

    with torch.no_grad():
        out_depth_maps = model.predict_depth_map(inputs)
    
    for i in range(cur_batch_size):
        cur_depth_map = out_depth_maps[i]
        cur_mask = gt_masks[i]
        depth_min = torch.min(cur_depth_map[cur_mask])
        depth_max = torch.max(cur_depth_map[cur_mask])

        if absolute_depth:
            # for absolute
            cur_depth_map[1. - cur_mask] = depth_max
            cur_depth_map = (cur_depth_map - depth_min) / (depth_max - depth_min)
        else:
            # for relative
            cur_depth_map[1. - cur_mask] = 1.
            cur_depth_map[cur_depth_map > 1] = 1.
            cur_depth_map[cur_depth_map < 0] = 0.
            

        cur_model_info = train_dataset.get_model_dict(idxs[i]) # category & model
        cur_viewid = viewids[i]

        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'])):
            os.mkdir(os.path.join(out_dir, cur_model_info['category']))
        
        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'])):
            os.mkdir(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model']))

        if not os.path.exists(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], pred_path)):
            os.mkdir(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], pred_path))
        
        # save png
        png_path = os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], pred_path, '%.2d_depth.png' % cur_viewid)
        visualize_data(cur_depth_map, 'img', png_path)
        # record range
        depth_range_path = os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], pred_path, 'depth_range.txt')
        
        if cur_viewid == 0:
            if os.path.exists(depth_range_path):
                os.remove(depth_range_path)
        with open(depth_range_path, mode='a') as f:
            print(depth_min.item(), depth_max.item(), 1.0, file=f)

    
    t1 = time.time()
    pbar.update(1)
    #print("\r finished: %d / %d in %d sec" % (it, batch_count, t1 - t0), flush=True)
pbar.close()    
