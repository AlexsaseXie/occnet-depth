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
import pandas as pd

# Arguments
parser = argparse.ArgumentParser(
    description='Generate depth predictions.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--out_dir', type=str, default='./data/ShapeNet.depth_pred.uresnet/', help='Output dir')
parser.add_argument('--out_folder_name', type=str, default='depth_pred', help='output folder name')
parser.add_argument('--batch_size', type=int, default=128) # 60 for hourglass
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--time_test', action='store_true', help='Time test')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = args.out_dir
print('out_dir:',out_dir)
pred_path = args.out_folder_name
batch_size = args.batch_size if not args.time_test else 1
print('input_type:', cfg['data']['input_type'])

if not os.path.exists(out_dir) and not args.time_test:
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
if not args.time_test:
    train_dataset = data.Shapes3dDataset_AllImgs(cfg['data']['path'], fields, split=None)
else:
    train_dataset = data.Shapes3dDataset_AllImgs(cfg['data']['path'], fields, split='test', n_views=1)

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

it = 0
batch_count = len(train_loader)

if 'absolute_depth' in cfg['data']:
    absolute_depth = cfg['data']['absolute_depth']
else:
    absolute_depth = True

print('absolute_depth:', absolute_depth)

from tqdm import tqdm
pbar = tqdm(total=batch_count)
time_dicts = []
for batch in train_loader:
    it += 1
    model.eval()

    inputs = batch.get('inputs').to(device)
    cur_batch_size = inputs.size(0)
    gt_masks = batch.get('inputs.mask').to(device).byte()

    idxs = batch.get('idx')
    viewids = batch.get('viewid')

    if args.time_test:
        single_t0 = time.time()

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

    pbar.update(1)
pbar.close()    

if args.time_test:
    if not os.path.exists('./timings/'):
        os.mkdir('./timings/')

    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle('./timings/depth_estimation_time_full.pkl')

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class id']).mean()
    time_df_class.to_pickle('./timings/depth_estimation_time_cls.pkl')

    time_df_class.loc['mean'] = time_df_class.mean()
    print('Timings: ', time_df_class)
