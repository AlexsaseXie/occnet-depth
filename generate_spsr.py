import os
import sys
import argparse
from multiprocessing import Pool
import subprocess
import numpy as np
import time
from functools import partial

DEBUG=1

parser = argparse.ArgumentParser(
    description='test'
)
parser.add_argument('--dataset_root', type=str, default='./data/ShapeNet.depth_pred.uresnet.origin_subdivision')
parser.add_argument('--depth_pointcloud_completion_folder', type=str, default='MSN_updated_space_carved_random_scale_4096')
parser.add_argument('--save_folder', type=str, default='spsr_updated')
parser.add_argument('--list', type=str, default=None)
parser.add_argument('--list_root', type=str, default='./data/ShapeNet.with_depth.10w10w/')
parser.add_argument('--list_file', type=str, default='updated_test.lst')
parser.add_argument('--spsr_mlx', type=str, default='./scripts/spsr.mlx')
parser.add_argument('--n_proc', type=int, default=15)
parser.add_argument('--test', action='store_true')

args = parser.parse_args()


def pcwrite(filename, xyzrgb, nxnynz=None, color=True, normal=False):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    if color:
        rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    if color:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
    if normal:
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
        ))

        if color:
            ply_file.write(" %d %d %d"%(
                rgb[i, 0], rgb[i, 1], rgb[i, 2],
            ))
        
        if normal:
            ply_file.write(" %f %f %f" % (
                nxnynz[i, 0], nxnynz[i, 1], nxnynz[i, 2]
            ))
        
        ply_file.write("\n")

def get_list():
    if args.list is not None:
        file_list = []
    else:
        file_list = []
        classes = os.listdir(args.dataset_root)
        for c in classes:
            class_root = os.path.join(args.dataset_root, c)
            if not os.path.exists(class_root):
                continue

            if args.list_file is not None:
                lst_path = os.path.join(args.list_root, c, args.list_file)
                with open(lst_path, 'r') as f:
                    modelnames = f.read().split('\n')
            else:
                modelnames = os.listdir(class_root)

            print('class %s modelname count: %d' % (c, len(modelnames)))
            for modelname in modelnames:
                target_file = os.path.join(class_root, modelname, args.depth_pointcloud_completion_folder, '00_pointcloud.npz')

                if os.path.exists(target_file):
                    file_list.append(target_file)

    print('Total files %d' % len(file_list))
    return file_list

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a)

def process_file(pc_file, output_folder=None):
    pc_folder = os.path.dirname(pc_file)
    model_root = os.path.dirname(pc_folder)

    data = np.load(pc_file)
    pc = data['pointcloud']

    if output_folder is None:
        save_root = os.path.join(model_root, args.save_folder)
        mkdir_p(save_root)
        ply_save_filename = os.path.join(save_root, '00.ply')
    else:
        ply_save_filename = os.path.join(output_folder, 'pc.ply')

    pcwrite(ply_save_filename, pc, color=False)

    if output_folder is None:
        ply_recon_save_filename = os.path.join(save_root, '00_recon.ply')
    else:
        ply_recon_save_filename = os.path.join(output_folder, 'pc_recon.ply')
        mkdir_p(output_folder)

    # meshlab
    sub_p = subprocess.Popen(
        ['meshlabserver', '-i', ply_save_filename, '-o', ply_recon_save_filename, 
        '-s', args.spsr_mlx], 
        stdout=subprocess.PIPE if DEBUG else open('/dev/null','w'), 
        stderr=subprocess.STDOUT
    )
    sub_p.wait()
    
    print('finish model %s' % pc_file)

def main():
    file_list = get_list()
    s_time = time.time()

    with Pool(args.n_proc) as p:
        p.map(partial(process_file, output_folder=None), file_list)

    print('Finish in %f s' % (time.time() - s_time))

def test():
    input_file = os.path.join(args.dataset_root, '02691156',
         '10155655850468db78d106ce0a280f87', args.depth_pointcloud_completion_folder, '00_pointcloud.npz')

    process_file(input_file, output_folder='./data/spsr_test/')

if __name__ == '__main__':
    if args.test:
        test()
    else:
        main()

    
        

