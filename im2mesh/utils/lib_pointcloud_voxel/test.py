from pointcloud_voxel import grid_points_query_range
import numpy as np
import sys
import os

sys.path.append('../')
sys.path.append('../../../')
sys.path.append('../../')
from visualize import visualize_voxels

DATA_FILE = [
    '/home3/xieyunwei/ShapeNet.SAL/02691156/10155655850468db78d106ce0a280f87/pointcloud_fps_N30000.npz',
    '/home3/xieyunwei/ShapeNet.SAL/02691156/1021a0914a7207aff927ed529ad90a11/pointcloud_fps_N30000.npz'
]

for df in DATA_FILE:
    print('Df: %s' % df)
    data_dict = np.load(df)
    pc = data_dict['points']

    voxel, calc_index, inside_index, outside_index = grid_points_query_range(pc, 256, 0.008, -0.55, 0.55)

    print('Finish calc')

    if not os.path.exists('./test_data/'):
        os.mkdir('test_data')

    voxel_surface = voxel[:,:,:,3] == 1
    #print('Finish get surface voxel')
    #visualize_voxels(voxel_surface, './test_data/surface_voxel.png')
    #print('Finish output pics')
    voxel_inside = voxel[:,:,:,3] == 2
    #visualize_voxels(voxel_inside,'./test_data/inside_voxel.png')

    print('Calc index:', calc_index.shape, calc_index[:50])
    print('Inside index:', inside_index.shape)
    print('Outside index:', outside_index.shape)

    print('V[0,0,0]:', voxel[0,0,0])
    print('V[10,10,10]:', voxel[10,10,10])
    print('V[49,0,0]:', voxel[49,0,0])

