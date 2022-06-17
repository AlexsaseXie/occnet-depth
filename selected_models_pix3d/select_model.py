import pickle
import os
import argparse

parser = argparse.ArgumentParser(
    description='Select Model'
)
parser.add_argument('--worst', action='store_true', help='Select worst cases')
parser.add_argument('--out_dir', type=str, default='selected_models_pix3d')
args = parser.parse_args()

worst = args.worst
ONet_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/onet/generation_pix3d/eval_meshes_full.pkl'
ONet_depth_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/phase2_depth_pointcloud_MSN_4096_pointnet2_local/generation_pix3d/eval_meshes_full.pkl'

def read_data(root, iou_newname):
    with open(root,'rb') as f:
        dataset = pickle.load(f)
    
    dataset.rename(columns={'iou (mesh)': iou_newname}, inplace=True)
    dataset = dataset[['class id', 'image name', 'model name', iou_newname]]

    return dataset

def mkdir_p(root):
    if not os.path.exists(root):
        os.mkdir(root) 

mkdir_p(args.out_dir)

ONet_dataset = read_data(ONet_root, 'iou_onet')
ONet_depth_dataset = read_data(ONet_depth_root, 'iou_ours')

m = ONet_dataset.merge(ONet_depth_dataset)

m['iou_delta'] = m.apply(lambda x: x['iou_ours'] - x['iou_onet'], axis=1)

if not worst:
    m = m.loc[(m['iou_ours'] >= 0.2)] #& (m['iou_onet'] >= 0.3)]

def select_iou_delta(d, m):
    return d.sort_values(by='iou_delta', ascending=worst)[:m]


t = m.groupby('class id').apply(select_iou_delta, 20)

print(t)

tmp = 'best' if not worst else 'worst'
t.to_pickle(os.path.join(args.out_dir,'selected_%s_full.pkl' % tmp))
t.to_csv(os.path.join(args.out_dir,'selected_%s.txt' % tmp), sep='\t', index=False)
