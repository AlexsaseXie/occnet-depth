import argparse
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
# TODO: do this better
sys.path.append('..')
from im2mesh.utils import binvox_rw, voxels
from im2mesh.utils.libmesh import check_mesh_contains
from im2mesh.utils.libkdtree import KDTree


parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')

parser.add_argument('--rotate_xz', type=float, default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

parser.add_argument('--pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=100000,
                    help='Size of point cloud.')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')

parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=500000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')
parser.add_argument('--points_subsample', type=int, default=50000,
                    help='subsampled points')
parser.add_argument('--bandwidth', type=float, default=0.2)


parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
    
def main(args):
    input_files = glob.glob(os.path.join(args.in_folder, '*.off'))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)


def process_path(in_path, args):
    in_file = os.path.basename(in_path)
    modelname = os.path.splitext(in_file)[0]
    filename = os.path.join(args.points_folder, modelname + '.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    mesh = trimesh.load(in_path, process=False)

    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '.off')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        if args.rotate_xz != 0:
            angle = args.rotate_xz / 180 * np.pi
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(R)

    # Expert various modalities
    if args.pointcloud_folder is not None:
        export_pointcloud(mesh, modelname, loc, scale, args)

    if args.voxels_folder is not None:
        export_voxels(mesh, modelname, loc, scale, args)

    if args.points_folder is not None:
        export_points(mesh, modelname, loc, scale, args)

    if args.mesh_folder is not None:
        export_mesh(mesh, modelname, loc, scale, args)


def export_pointcloud(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.pointcloud_folder,
                            modelname + '.npz')
    if not args.overwrite and os.path.exists(filename):
        print('Pointcloud already exist: %s' % filename)
        return

    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)


def export_voxels(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot create voxelization.' % modelname)
        return

    filename = os.path.join(args.voxels_folder, modelname + '.binvox')

    if not args.overwrite and os.path.exists(filename):
        print('Voxels already exist: %s' % filename)
        return

    res = args.voxels_res
    voxels_occ = voxels.voxelize(mesh, res)

    voxels_out = binvox_rw.Voxels(voxels_occ, (res,) * 3,
                                  translate=loc, scale=scale,
                                  axis_order='xyz')
    print('Writing voxels: %s' % filename)
    with open(filename, 'bw') as f:
        voxels_out.write(f)


def sample_points_sdf(num_sample, points, sdf, occupancies, bandwidth=0.2):
    percentages = [[-3. * bandwidth,  -1. * bandwidth, int(num_sample * 0.2)],
                  [-1. * bandwidth,   0,               int(num_sample * 0.3)],
                  [0,                 bandwidth,       int(num_sample * 0.3)],
                  [bandwidth,         bandwidth * 3.,  int(num_sample * 0.2)]]

    sampled_points = []
    sampled_sdf = []
    sampled_occ = []
    for i in range(len(percentages)):
        ind = np.argwhere((sdf >= percentages[i][0]) & (sdf < percentages[i][1]))
        if len(ind) < percentages[i][2]:
            print("lack in cate ",i,": %d/%d" % (len(ind), percentages[i][2]) )
            if i < len(percentages)-1:
                percentages[i+1][2] += percentages[i][2] - len(ind)
                percentages[i][2] = len(ind)
            else:
                # last
                ind = np.argwhere((sdf >= percentages[i][0]))
                if len(ind) < percentages[i][2]:
                    ind = np.argwhere((sdf >= 0))
        else:
            print("filled cate ",i)

        if len(ind) == 0:
            #print("len(ind) == 0 for cate ", i)
            continue

        choice = np.random.choice(range(len(ind)), size=percentages[i][2], replace=False)
        #choice = np.s_[0:percentages[i][2]]
        choice_ind = ind[choice]
        sampled_points.append(points[choice_ind])
        sampled_sdf.append(sdf[choice_ind])
        sampled_occ.append(occupancies[choice_ind])
    
    sampled_points = np.concatenate(sampled_points, axis=0)
    sampled_sdf = np.concatenate(sampled_sdf, axis=0)
    sampled_occ = np.concatenate(sampled_occ, axis=0)
    
    assert sampled_points.shape[0] == num_sample
    assert sampled_sdf.shape[0] == num_sample
    assert sampled_occ.shape[0] == num_sample

    return sampled_points, sampled_sdf, sampled_occ


def export_points(mesh, modelname, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return

    filename = os.path.join(args.points_folder, modelname + '.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    ratio_world = mesh.volume / ((1+args.points_padding) ** 3)
    ratio_bbox = mesh.volume / mesh.bounding_box.volume
    print('Volume ratio, world:%.4f; bbox:%.4f' % (ratio_world, ratio_bbox) )

    N = args.points_subsample
    bandwidth = args.bandwidth
    N_bandwidth = int(N * 0.7)

    # for surface bandwidth points:
    n_points_surface = int(N_bandwidth * 5)
    points_surface, points_index = mesh.sample(n_points_surface, return_index=True)
    points_normal = mesh.face_normals[points_index]
    points_offset = bandwidth * (np.random.rand(n_points_surface, 1) * 2. - 1.)
    points_surface += points_offset * points_normal

    # for other points
    #ratio_min = min(ratio_world, 1. - ratio_world)
    #N_other = int(N * 0.15) 

    n_points_uniform = args.points_size

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)

    points = np.concatenate([points_uniform, points_surface], axis=0)
    occupancies = check_mesh_contains(mesh, points)

    # calculate distance
    mesh_sampled_points = mesh.sample(count=100000)
    kdtree = KDTree(mesh_sampled_points)
    dist,_ = kdtree.query(points)

    sdf = (occupancies * (-2.) + 1.).astype(np.float32) * dist

    # Compress
    dtype = np.float32
    points = points.astype(dtype)
    sdf = sdf.astype(dtype)

    #print("SDF: min", sdf.min(), "max:", sdf.max(), "avg:", sdf.mean())
    #for i in range(10):
    #    percent = i * 10
    #    print("SDF: below %d" % percent, np.percentile(sdf, percent))

    if args.points_subsample != 0:
        # subsample
        points, sdf, occupancies = sample_points_sdf(args.points_subsample, points, sdf, occupancies, bandwidth=args.bandwidth)
        
    points = np.squeeze(points)
    sdf = np.squeeze(sdf)
    occupancies = np.squeeze(occupancies)

    # save     
    print('Writing points: %s' % filename)
    np.savez(filename, points=points, sdf=sdf, occupancies=occupancies,
             loc=loc, scale=scale)


def export_mesh(mesh, modelname, loc, scale, args):
    filename = os.path.join(args.mesh_folder, modelname + '.off')    
    if not args.overwrite and os.path.exists(filename):
        print('Mesh already exist: %s' % filename)
        return
    print('Writing mesh: %s' % filename)
    mesh.export(filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
