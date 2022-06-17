import pickle
import os
import argparse

parser = argparse.ArgumentParser(
    description='Select Model'
)
parser.add_argument('--worst', action='store_true', help='Select worst cases')
parser.add_argument('--out_dir', type=str, default='selected_models')
args = parser.parse_args()

worst = args.worst
DISN_root = '/home2/xieyunwei/DISN/checkpoint/224_retrain/direct_pad_test_objs/65_0.0/eval_meshes_full_tsdf0.002.pkl'
IMNet_root = '/home2/xieyunwei/IM-NET-pytorch/samples/im_svr_224_all_out/eval_meshes_full_tsdf0.002.pkl'
ONet_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/onet/generation_simplify/eval_meshes_full_tsdf0.002.pkl'
ONet_depth_root = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/phase2_depth_pointcloud_MSN_4096_pointnet2_4layers_version2(dropout)_local_clean/generation_space_carved/eval_meshes_full_tsdf0.002.pkl'

def read_data(root, iou_newname):
    with open(root,'rb') as f:
        dataset = pickle.load(f)
    
    dataset.rename(columns={'iou (mesh)': iou_newname}, inplace=True)
    dataset = dataset[['class id', 'modelname', iou_newname]]

    return dataset

def mkdir_p(root):
    if not os.path.exists(root):
        os.mkdir(root) 

mkdir_p(args.out_dir)

DISN_dataset = read_data(DISN_root, 'iou_disn')
IMNet_dataset = read_data(IMNet_root, 'iou_imnet')
ONet_dataset = read_data(ONet_root, 'iou_onet')
ONet_depth_root = read_data(ONet_depth_root, 'iou_ours')

m = DISN_dataset.merge(IMNet_dataset)
m = m.merge(ONet_dataset)
m = m.merge(ONet_depth_root)

m['iou_delta'] = m.apply(lambda x: x['iou_ours'] * 3 - x['iou_disn'] - x['iou_imnet'] - x['iou_onet'], axis=1)

if not worst:
    m = m.loc[(m['iou_disn'] >= 0.4) & (m['iou_imnet'] >= 0.4) & (m['iou_onet'] >= 0.4) & (m['iou_ours'] >= 0.4)]

def select_iou_delta(d, m):
    return d.sort_values(by='iou_delta', ascending=worst)[:m]


t = m.groupby('class id').apply(select_iou_delta, 20)

print(t)

tmp = 'best' if not worst else 'worst'
t.to_pickle(os.path.join(args.out_dir,'selected_%s_full.pkl' % tmp))
t.to_csv(os.path.join(args.out_dir,'selected_%s.txt' % tmp), sep='\t', index=False)
