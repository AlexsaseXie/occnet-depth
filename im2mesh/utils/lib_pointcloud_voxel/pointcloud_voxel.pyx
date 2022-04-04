# distutils: language = c++
cimport cython
from cython.operator cimport dereference as dref
from libcpp.vector cimport vector
from libcpp.map cimport map
from libc.math cimport isnan, NAN
import numpy as np

cdef extern from "pointcloud_voxel_c.h":
    void find_nearby_grid_points(float * pointcloud, int pointcloud_N, int N, \
        float * grid_points, long * calculate_grid_points_index, int * calculate_N, \
        long * inside_grid_points_index, int * inside_grid_points_N, \
        long * outside_grid_points_index, int * outside_grid_points_N, int tolerance_K, \
        float low, float high);

    void find_nearby_grid_points_range(float * pointcloud, int pointcloud_N, int N, \
        float * grid_points, long * calculate_grid_points_index, int * calculate_N, \
        long * inside_grid_points_index, int * inside_grid_points_N, \
        long * outside_grid_points_index, int * outside_grid_points_N, float tolerance_K, \
        float low, float high);

def grid_points_query(float[:,::1] pointcloud, int N, int toler_K=1, float low=-0.5, float high=0.5):
    '''
        Input: pointcloud (N, 3)
            N 
        Output:
            grid points (N, N, N, 3)
            calculate_index (T, 3) x,y,z index on grid points
            outside_index (M, 3) 
            inside_index (N**3 - T - M, 3)
    '''
    cdef int max_len = N * N * N

    cdef int pointcloud_N = pointcloud.shape[0]
    cdef float* pointcloud_pt = &(pointcloud[0,0])

    grid_points = np.zeros((N, N, N, 4), dtype=np.float32)
    cdef float[:,:,:,::1] grid_points_view = grid_points
    cdef float* grid_points_pt = &(grid_points_view[0,0,0,0])

    calculate_grid_points_index = np.zeros((max_len, 3), dtype=np.int)
    cdef long[:,::1] calculate_grid_points_index_view = calculate_grid_points_index
    cdef long* calculate_grid_points_index_pt = &(calculate_grid_points_index_view[0,0])

    cdef int calculate_N = 0
    cdef int* calculate_N_pt = &calculate_N

    inside_points_index = np.zeros((max_len, 3), dtype=np.int)
    cdef long[:,::1] inside_points_index_view = inside_points_index
    cdef long* inside_points_index_pt = &(inside_points_index_view[0,0])

    cdef int inside_N = 0
    cdef int* inside_N_pt = &inside_N

    outside_points_index = np.zeros((max_len, 3), dtype=np.int)
    cdef long[:,::1] outside_points_index_view = outside_points_index
    cdef long* outside_points_index_pt = &(outside_points_index_view[0,0])

    cdef int outside_N = 0
    cdef int* outside_N_pt = &outside_N


    find_nearby_grid_points(pointcloud_pt, pointcloud_N, N, grid_points_pt, 
        calculate_grid_points_index_pt, calculate_N_pt, 
        inside_points_index_pt, inside_N_pt,
        outside_points_index_pt, outside_N_pt, 
        toler_K, low, high)

    calculate_grid_points_index = calculate_grid_points_index[:calculate_N, :]
    inside_points_index = inside_points_index[:inside_N, :]
    outside_points_index = outside_points_index[:outside_N, :]
    return grid_points, calculate_grid_points_index, inside_points_index, outside_points_index


def grid_points_query_range(float[:,::1] pointcloud, int N, float toler_K=0.02, float low=-0.5, float high=0.5):
    '''
        Input: pointcloud (N, 3)
            N 
        Output:
            grid points (N, N, N, 3)
            calculate_index (T, 3) x,y,z index on grid points
            outside_index (M, 3) 
            inside_index (N**3 - T - M, 3)
    '''
    cdef int max_len = N * N * N

    cdef int pointcloud_N = pointcloud.shape[0]
    cdef float* pointcloud_pt = &(pointcloud[0,0])

    grid_points = np.zeros((N, N, N, 4), dtype=np.float32)
    cdef float[:,:,:,::1] grid_points_view = grid_points
    cdef float* grid_points_pt = &(grid_points_view[0,0,0,0])

    calculate_grid_points_index = np.zeros((max_len, 3), dtype=np.int)
    cdef long[:,::1] calculate_grid_points_index_view = calculate_grid_points_index
    cdef long* calculate_grid_points_index_pt = &(calculate_grid_points_index_view[0,0])

    cdef int calculate_N = 0
    cdef int* calculate_N_pt = &calculate_N

    inside_points_index = np.zeros((max_len, 3), dtype=np.int)
    cdef long[:,::1] inside_points_index_view = inside_points_index
    cdef long* inside_points_index_pt = &(inside_points_index_view[0,0])

    cdef int inside_N = 0
    cdef int* inside_N_pt = &inside_N

    outside_points_index = np.zeros((max_len, 3), dtype=np.int)
    cdef long[:,::1] outside_points_index_view = outside_points_index
    cdef long* outside_points_index_pt = &(outside_points_index_view[0,0])

    cdef int outside_N = 0
    cdef int* outside_N_pt = &outside_N


    find_nearby_grid_points_range(pointcloud_pt, pointcloud_N, N, grid_points_pt, 
        calculate_grid_points_index_pt, calculate_N_pt, 
        inside_points_index_pt, inside_N_pt,
        outside_points_index_pt, outside_N_pt, 
        toler_K, low, high)

    calculate_grid_points_index = calculate_grid_points_index[:calculate_N, :]
    inside_points_index = inside_points_index[:inside_N, :]
    outside_points_index = outside_points_index[:outside_N, :]
    return grid_points, calculate_grid_points_index, inside_points_index, outside_points_index



    
    