cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array()



cdef extern from "fusion.h":
  cdef cppclass Views:
    Views()
    int n_views_;
    float* depthmaps_;
    int rows_;
    int cols_;
    float* Ks_;
    float* Rs_;
    float* Ts_;

  cdef cppclass Volume:
    Volume()
    int channels_;
    int depth_;
    int height_;
    int width_;
    float* data_;

  cdef cppclass Points:
    Points()
    int n_pts_;
    int c_dim_;
    float * data_;


  void fusion_projectionmask_gpu(const Views& views, float vx_size, bool unknown_is_free, Volume& vol);
  void fusion_occupancy_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);
  void fusion_tsdfmask_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);
  void fusion_tsdf_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);
  void fusion_tsdf_hist_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, bool unobserved_is_occupied, Volume& vol);
  void fusion_tsdf_strict_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);
  void fusion_tsdf_range_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);
  void fusion_hist_zach_tvl1_gpu(const Volume& hist, bool hist_on_gpu, float truncation, float lambda_param, int iterations, Volume& vol);
  void fusion_zach_tvl1_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, float lambda_param, int iterations, Volume& vol);
  void fusion_inside_gpu(const Views &views, int n_pts, Points & points);
  void fusion_view_tsdf_estimation(const Views &views, Points &query, float truncated_distance);
  void fusion_view_pc_tsdf_estimation(const Points& pointcloud, const Views& views, Points &query, float truncated_distance, int aggregate_type);
  void fusion_view_pc_tsdf_estimation_var(const Points& pointcloud, const Views& views, Points &query, float truncated_distance, int aggregate_type);
  void fusion_nn_pc(const Points& pointcloud, Points &query);

cdef class PyViews:
  cdef Views views
  # need to keep reference, otherwise it could get garbage collected
  cdef float[:,:,::1] depthmaps_ 
  cdef float[:,:,::1] Ks_
  cdef float[:,:,::1] Rs_
  cdef float[:,::1] Ts_

  def __init__(self, float[:,:,::1] depthmaps, float[:,:,::1] Ks, float[:,:,::1] Rs, float[:,::1] Ts):
    cdef int n = depthmaps.shape[0]
    if n != Ks.shape[0]:
      raise Exception('number of depthmaps and Ks differ')
    if n != Rs.shape[0]:
      raise Exception('number of depthmaps and Rs differ')
    if n != Ts.shape[0]:
      raise Exception('number of depthmaps and Ts differ')

    if Ks.shape[1] != 3 or Ks.shape[2] != 3:
      raise Exception('Ks have to be nx3x3')
    if Rs.shape[1] != 3 or Rs.shape[2] != 3:
      raise Exception('Rs have to be nx3x3')
    if Ts.shape[1] != 3:
      raise Exception('Ts have to be nx3')

    self.depthmaps_ = depthmaps
    self.Ks_ = Ks
    self.Rs_ = Rs
    self.Ts_ = Ts
    
    self.views.depthmaps_ = &(depthmaps[0,0,0])  
    self.views.n_views_ = depthmaps.shape[0]
    self.views.rows_ = depthmaps.shape[1]
    self.views.cols_ = depthmaps.shape[2]
    self.views.Ks_ = &(Ks[0,0,0])  
    self.views.Rs_ = &(Rs[0,0,0]) 
    self.views.Ts_ = &(Ts[0,0]) 


cdef class PyVolume:
  cdef Volume vol

  def __init__(self, float[:,:,:,::1] data):
    self.vol = Volume()
    self.vol.data_ = &(data[0,0,0,0])
    self.vol.channels_ = data.shape[0]
    self.vol.depth_ = data.shape[1]
    self.vol.height_ = data.shape[2]
    self.vol.width_ = data.shape[3]


cdef class PyPoints:
  cdef Points points

  def __init__(self, float[:,::1] data):
    self.points = Points()
    self.points.data_ = &(data[0,0])
    self.points.n_pts_ = data.shape[0]
    self.points.c_dim_ = data.shape[1]


def projmask_gpu(PyViews views, int depth, int height, int width, float vx_size, bool unknown_is_free):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_projectionmask_gpu(views.views, vx_size, unknown_is_free, py_vol.vol)
  return vol
def occupancy_gpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_occupancy_gpu(views.views, vx_size, truncation, unknown_is_free, py_vol.vol)
  return vol

def tsdfmask_gpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdfmask_gpu(views.views, vx_size, truncation, unknown_is_free, py_vol.vol)
  return vol

def tsdf_gpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdf_gpu(views.views, vx_size, truncation, unknown_is_free, py_vol.vol)
  return vol

def tsdf_strict_gpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdf_strict_gpu(views.views, vx_size, truncation, unknown_is_free, py_vol.vol)
  return vol

def tsdf_range_gpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdf_range_gpu(views.views, vx_size, truncation, unknown_is_free, py_vol.vol)
  return vol

def tsdf_hist_gpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, float[::1] bins, bool unobserved_is_occupied=True):
  cdef int n_bins = bins.shape[0]
  vol = np.empty((n_bins, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdf_hist_gpu(views.views, vx_size, truncation, unknown_is_free, &(bins[0]), n_bins, unobserved_is_occupied, py_vol.vol)
  return vol



def zach_tvl1_hist(float[:,:,:,::1] hist, float truncation, float lambda_param, int iterations, init=None):
  vol = np.zeros((1, hist.shape[1], hist.shape[2], hist.shape[3]), dtype=np.float32)
  if init is not None:
    vol[...] = init.reshape(vol.shape)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  cdef PyVolume py_hist = PyVolume(hist)
  fusion_hist_zach_tvl1_gpu(py_hist.vol, False, truncation, lambda_param, iterations, py_vol.vol)
  return vol

def zach_tvl1(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, float[::1] bins, float lambda_param, int iterations, init=None):
  cdef int n_bins = bins.shape[0]
  vol = np.zeros((1, depth, height, width), dtype=np.float32)
  if init is not None:
    vol[...] = init.reshape(vol.shape)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_zach_tvl1_gpu(views.views, vx_size, truncation, unknown_is_free, &(bins[0]), n_bins, lambda_param, iterations, py_vol.vol)
  return vol

def judge_inside(PyViews views, points):
  cdef int n_pts = points.shape[0]
  cdef int c_dim = points.shape[1]
  return_points = np.zeros((n_pts, c_dim + 2), dtype=np.float32)
  return_points[:,:c_dim] = points[:,:c_dim]

  cdef float[:,::1] return_points_view = return_points
  cdef PyPoints py_points = PyPoints(return_points_view)
  fusion_inside_gpu(views.views, n_pts, py_points.points)
  return return_points

def judge_inside_with_threshold(PyViews views, points, outside_min_view):
  cdef int n_pts = points.shape[0]
  cdef int c_dim = points.shape[1]
  return_points = np.zeros((n_pts, c_dim + 2), dtype=np.float32)
  return_points[:,:c_dim] = points[:,:c_dim]

  cdef float[:,::1] return_points_view = return_points
  cdef PyPoints py_points = PyPoints(return_points_view)
  fusion_inside_gpu(views.views, n_pts, py_points.points)

  points_outside_view_count = return_points[:,c_dim]
  occ = ~(points_outside_view_count > outside_min_view - 0.5)
  return occ

def view_tsdf_estimation(PyViews views, points, float truncation):
  cdef int n_pts = points.shape[0]
  cdef int c_dim = points.shape[1]

  return_points = np.zeros((n_pts, c_dim + 1), dtype=np.float32)
  return_points[:,:c_dim] = points[:,:c_dim]
  cdef float[:,::1] return_points_view = return_points
  cdef PyPoints py_points = PyPoints(return_points_view)

  fusion_view_tsdf_estimation(views.views, py_points.points, truncation)

  tsdf = return_points[:,c_dim - 1]
  return tsdf

def view_pc_tsdf_estimation(PyViews views, pointcloud, points, float truncation, aggregate):
  cdef float[:,::1] pointcloud_view = pointcloud
  cdef PyPoints py_pointcloud = PyPoints(pointcloud_view)

  cdef int n_pts = points.shape[0]
  cdef int c_dim = points.shape[1]

  return_points = np.zeros((n_pts, c_dim + 1), dtype=np.float32)
  return_points[:,:c_dim] = points[:,:c_dim]
  cdef float[:,::1] return_points_view = return_points
  cdef PyPoints py_points = PyPoints(return_points_view)

  cdef int aggregate_type = 0

  if aggregate == 'min':
    aggregate_type = 0
  elif aggregate == 'mean':
    aggregate_type = 1
  else:
    raise NotImplementedError

  fusion_view_pc_tsdf_estimation(py_pointcloud.points, views.views, py_points.points, truncation, aggregate_type)
  tsdf = return_points[:,c_dim]

  return tsdf

def view_pc_tsdf_estimation_var(PyViews views, pointcloud, points, float truncation, aggregate):
  cdef float[:,::1] pointcloud_view = pointcloud
  cdef PyPoints py_pointcloud = PyPoints(pointcloud_view)

  cdef int n_pts = points.shape[0]
  cdef int c_dim = points.shape[1]

  return_points = np.zeros((n_pts, c_dim + 1), dtype=np.float32)
  return_points[:,:c_dim] = points[:,:c_dim]
  cdef float[:,::1] return_points_view = return_points
  cdef PyPoints py_points = PyPoints(return_points_view)

  cdef int aggregate_type = 0

  if aggregate == 'min':
    aggregate_type = 0
  elif aggregate == 'mean':
    aggregate_type = 1
  else:
    raise NotImplementedError

  fusion_view_pc_tsdf_estimation_var(py_pointcloud.points, views.views, py_points.points, truncation, aggregate_type)
  tsdf = return_points[:,c_dim]

  return tsdf

def pc_nn(pointcloud, points):
  cdef float[:,::1] pointcloud_view = pointcloud
  cdef PyPoints py_pointcloud = PyPoints(pointcloud_view)

  cdef int n_pts = points.shape[0]
  cdef int c_dim = points.shape[1]

  return_points = np.zeros((n_pts, c_dim + 2), dtype=np.float32)
  return_points[:,:c_dim] = points[:,:c_dim]
  cdef float[:,::1] return_points_view = return_points
  cdef PyPoints py_points = PyPoints(return_points_view)

  fusion_nn_pc(py_pointcloud.points, py_points.points)
  dis = return_points[:, c_dim]
  idx = np.floor(return_points[:, c_dim + 1]).astype(np.int)

  return dis, idx
