import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from im2mesh import config, data
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils


parser = argparse.ArgumentParser(
    description='Evaluate mesh algorithms.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--output_file_name', type=str, default='pointcloud_fps')
parser.add_argument('--N1', type=int, default=16384)
parser.add_argument('--N2', type=int, default=0)
parser.add_argument('--N3', type=int, default=0)
parser.add_argument('--out_dir', type=str, default=None)
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

# Get configuration and basic arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

dataset_folder = cfg['data']['path']
pointcloud_file = cfg['data']['pointcloud_file']
def get_fields():
    fields = {}
    fields['inputs'] = data.PointCloudField(pointcloud_file, with_transforms=True)
    fields['idx'] = data.IndexField()
    return fields

fields = get_fields()


# Dataset
do_categories = ['03001627',
'02958343',
'04256520',
'02691156',
'03636649',
'04401088',
'04530566',
'03691459',
'02933112',
'04379243',
'03211117',
'02828884',
'04090263'
]
train_dataset = data.Shapes3dDataset(dataset_folder, fields, split=cfg['data']['train_split'], categories=do_categories)
test_dataset = data.Shapes3dDataset(dataset_folder, fields, split=cfg['data']['test_split'], categories=do_categories)
val_dataset = data.Shapes3dDataset(dataset_folder, fields, split=cfg['data']['val_split'], categories=do_categories)

print('train len: %d, val len: %d, test len: %d' % (len(train_dataset), len(val_dataset), len(test_dataset)))
print('total len: %d' % (len(train_dataset) + len(val_dataset) + len(test_dataset)))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

Ns = [args.N1]
if args.N2 != 0:
    Ns.append(args.N2)
if args.N3 != 0:
    Ns.append(args.N3)

if args.out_dir is None:
    out_dir = dataset_folder
else:
    out_dir = args.out_dir
print('Out dir:', out_dir)

def process_dataset(dataset, dataloader):
    for batch in tqdm(dataloader):
        raw_pointcloud = batch.get('inputs')
        raw_normal = batch.get('inputs.normals')
        loc = batch.get('inputs.loc')
        scale = batch.get('inputs.scale')

        cur_batch_size = raw_pointcloud.size(0)
        idxs = batch.get('idx')

        with torch.no_grad(): 
            pointcloud = raw_pointcloud.to(device)
            normal = raw_normal.to(device)
            pointcloud_flipped = pointcloud.transpose(1, 2).contiguous()
            normal_flipped = normal.transpose(1, 2).contiguous()

        for N in Ns:
            with torch.no_grad():
                pointcloud_idx = pointnet2_utils.furthest_point_sample(pointcloud, N)
                
                pointcloud_processed = pointnet2_utils.gather_operation(pointcloud_flipped, pointcloud_idx).transpose(1, 2).contiguous()
                normal_processed = pointnet2_utils.gather_operation(normal_flipped, pointcloud_idx).transpose(1, 2).contiguous()

            for i in range(cur_batch_size):
                cur_pointcloud = pointcloud_processed[i].cpu().numpy()
                cur_normal = normal_processed[i].cpu().numpy()
                cur_loc = loc[i].numpy()
                cur_scale = scale[i].numpy()

                cur_model_info = dataset.get_model_dict(idxs[i]) # category & model

                if not os.path.exists(os.path.join(out_dir, cur_model_info['category'])):
                    os.mkdir(os.path.join(out_dir, cur_model_info['category']))
                
                if not os.path.exists(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'])):
                    os.mkdir(os.path.join(out_dir, cur_model_info['category'], cur_model_info['model']))
                
                output_file_name = args.output_file_name + '_N%2d.npz'
                save_pc_path = os.path.join(out_dir, cur_model_info['category'], cur_model_info['model'], 
                    output_file_name % N)

                np.savez(save_pc_path, points=cur_pointcloud, normals=cur_normal, loc=cur_loc, scale=cur_scale)    

print('Split train:')
process_dataset(train_dataset, train_loader)
print('Split val:')
process_dataset(val_dataset, val_loader)
print('Split test:')
process_dataset(test_dataset, test_loader)