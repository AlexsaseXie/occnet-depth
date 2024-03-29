cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array() 


cdef extern from "offscreen.h":
  void renderDepthMesh(double *FM, int fNum, double *VM, int vNum, double *CM, double *intrinsics, int *imgSizeV, double *zNearFarV, unsigned char * imgBuffer, float *depthBuffer, bool *maskBuffer, double linewidth, bool coloring);


def render(double[:,::1] vertices, double[:,::1] faces, double[::1] cam_intr, double[::1] znf, int[::1] img_size):
  if vertices.shape[0] != 3:
    raise Exception('vertices must be a 3xM double array')
  if faces.shape[0] != 3:
    raise Exception('faces must be a 3xM double array')
  if cam_intr.shape[0] != 4:
    raise Exception('cam_intr must be a 4x1 double vector')
  if img_size.shape[0] != 2:
    raise Exception('img_size must be a 2x1 int vector')

  cdef double* VM = &(vertices[0,0])
  cdef int vNum = vertices.shape[1]
  cdef double* FM = &(faces[0,0])
  cdef int fNum = faces.shape[1]
  cdef double* intrinsics = &(cam_intr[0])
  cdef double* zNearVarV = &(znf[0])
  cdef int* imgSize = &(img_size[0])

  cdef bool coloring = False
  cdef double* CM = NULL

  depth = np.empty((img_size[1], img_size[0]), dtype=np.float32)
  mask  = np.empty((img_size[1], img_size[0]), dtype=np.uint8)
  img   = np.empty((3, img_size[1], img_size[0]), dtype=np.uint8)
  cdef float[:,::1] depth_view = depth
  cdef unsigned char[:,::1] mask_view = mask
  cdef unsigned char[:,:,::1] img_view = img
  cdef float* depthBuffer = &(depth_view[0,0])
  cdef bool* maskBuffer = <bool*> &(mask_view[0,0])
  cdef unsigned char* imgBuffer = &(img_view[0,0,0])

  renderDepthMesh(FM, fNum, VM, vNum, CM, intrinsics, imgSize, zNearVarV, imgBuffer, depthBuffer, maskBuffer, 0, coloring);

  return depth.T, mask.T, img.transpose((2,1,0))


# new render function
cdef extern from "offscreen_new.h":
  void render_mesh( \
    float *vertex_array, float *color_array, float *normal_array, int fM, \
    bool use_color, \
    float *camera_position, int T, \
    float *intrinsics, int *imgSizeV, float *zNearFarV, \
    unsigned char * imageBuffer, float *depthBuffer, bool *maskBuffer, \
    float *normalBuffer, float *vertexBuffer, float* viewMatBuffer \
  );

  void select_vertex( \
    float * vertexBuffer, float * normalBuffer, \
    int * imgSizeV, \
    int fM, int T, \
    float *point_cloud, int *point_cloud_size, \
    float *face_normal_buffer, \
    int * stats \
  );

  void select_faces( \
    float * vertexBuffer, float * normalBuffer, \
    int * imgSizeV, \
    int fM, int T, \
    bool *face_visible_buffer, \
    float *face_normal_buffer,  \
    int *stats \
  );



def render_new(float[:,::1] vertex_array, float[:,::1] color_array, float[:,::1] normal_array, \
  float[:, ::1] cam_position, float[::1] cam_intr, float[::1] znf, int[::1] img_size):
  # input infos
  if vertex_array.shape[1] != 3:
    raise Exception('vertices must be a [N, 3] float array')
  if normal_array.shape[1] != 3:
    raise Exception('normals must be a [N, 3] float array')
  
  cdef int vertex_num = vertex_array.shape[0]
  assert vertex_num % 3 == 0
  cdef int fM = vertex_num // 3 # face num = v_count / 3
  #print("face count:", fM)
  assert normal_array.shape[0] == vertex_num

  cdef float* vertex_arr = &(vertex_array[0,0])
  cdef float* normal_arr = &(normal_array[0,0])

  cdef bool use_color = False
  cdef float* color_arr = NULL
  if (color_array.ndim == 2) and (color_array.shape[1] == 3) and (color_array.shape[0] != 0):
    # N * 3 color
    assert color_array.shape[0] == vertex_num
    use_color = True
    color_arr = &(color_array[0,0]) 
    
   
  if cam_position.shape[1] != 3:
    raise Exception('cam_postion must be a [T, 3] float vector')

  cdef int T = cam_position.shape[0]
  cdef float* cam_pos = &(cam_position[0,0])

  if cam_intr.shape[0] != 4:
    raise Exception('cam_intr must be a [4] float vector')
  if img_size.shape[0] != 2:
    raise Exception('img_size must be a [2] int vector')
  if znf.shape[0] != 2:
    raise Exception('znf must be a [2] float vector')

  cdef float* intrinsics = &(cam_intr[0])
  cdef float* zNearFarV = &(znf[0])
  cdef int* imgSizeV = &(img_size[0])    # [height, width]

  depth = np.empty((T, img_size[0], img_size[1]), dtype=np.float32)
  mask  = np.empty((T, img_size[0], img_size[1]), dtype=np.uint8)
  img   = np.empty((T, img_size[0], img_size[1], 4), dtype=np.uint8)
  normal = np.empty((T, img_size[0], img_size[1], 4), dtype=np.float32)
  vertex = np.empty((T, img_size[0], img_size[1], 4), dtype=np.float32)
  view_mat = np.empty((T, 4, 4), dtype=np.float32)
  cdef float[:,:,::1] depth_view = depth
  cdef unsigned char[:,:,::1] mask_view = mask
  cdef unsigned char[:,:,:,::1] img_view = img
  cdef float[:,:,:,::1] normal_view = normal
  cdef float[:,:,:,::1] vertex_view = vertex
  cdef float[:,:,::1] view_mat_view = view_mat

  cdef float* depthBuffer = &(depth_view[0,0,0])
  cdef bool* maskBuffer = <bool*> &(mask_view[0,0,0])
  cdef unsigned char* imageBuffer = &(img_view[0,0,0,0])
  cdef float * normalBuffer = &(normal_view[0,0,0,0])
  cdef float * vertexBuffer = &(vertex_view[0,0,0,0])
  cdef float * viewMatBuffer = &(view_mat_view[0,0,0])

  render_mesh( \
    vertex_arr, color_arr, normal_arr, fM, \
    use_color, \
    cam_pos, T, \
    intrinsics, imgSizeV, zNearFarV, \
    imageBuffer, depthBuffer, maskBuffer, \
    normalBuffer, vertexBuffer, viewMatBuffer \
  );

  #print('Depth:', depth.shape, depth)
  #print('Normal:', normal.shape, normal)
  return depth, mask, img, normal, vertex, view_mat

def select_vertex_from_buffer(float[:,:,:,::1] normal, float[:,:,:,::1] vertex, int fM, int[::1] img_size, int max_pointcloud_size = 500000):
  '''
    stats[0] = double_sided_face_count;
    stats[1] = bad_face_count;
    stats[2] = total_visible_face_count;
  '''
  cdef int pointcloud_size = max_pointcloud_size
  cdef int * pointcloud_size_pt = &pointcloud_size

  cdef int T = normal.shape[0]
  cdef int* imgSizeV = &(img_size[0])
  cdef float * vertexBuffer = &(vertex[0,0,0,0])
  cdef float * normalBuffer = &(normal[0,0,0,0]) 

  pointcloud = np.empty((pointcloud_size, 6), dtype=np.float32)
  cdef float[:,::1] pointcloud_view = pointcloud
  cdef float * point_cloud_buffer = &(pointcloud_view[0,0])

  stats = np.empty((3), dtype=np.int32)
  cdef int[::1] stats_view = stats
  cdef int * stats_buffer = &(stats_view[0])

  face_normal = np.empty((fM, 3), dtype=np.float32)
  cdef float[:,::1] face_normal_view = face_normal
  cdef float * face_normal_buffer = &(face_normal_view[0,0])

  # select
  select_vertex( \
    vertexBuffer, normalBuffer, \
    imgSizeV, \
    fM, T, \
    point_cloud_buffer, pointcloud_size_pt, \
    face_normal_buffer, \
    stats_buffer);
  
  # slice
  pointcloud = pointcloud[:pointcloud_size, :]

  return pointcloud, face_normal, stats

def select_faces_from_buffer(float[:,:,:,::1] normal, float[:,:,:,::1] vertex, int fM, int[::1] img_size):
  '''
    stats[0] = double_sided_face_count;
    stats[1] = bad_face_count;
    stats[2] = total_visible_face_count;
  '''
  cdef int T = normal.shape[0]
  cdef int* imgSizeV = &(img_size[0])
  cdef float * vertexBuffer = &(vertex[0,0,0,0])
  cdef float * normalBuffer = &(normal[0,0,0,0]) 

  face_visible = np.empty((fM), dtype=np.bool)
  cdef bool[::1] face_visible_view = face_visible
  cdef bool * face_visible_buffer = &(face_visible_buffer[0])

  stats = np.empty((3), dtype=np.int32)
  cdef int[::1] stats_view = stats
  cdef int * stats_buffer = &(stats_view[0])

  face_normal = np.empty((fM, 3), dtype=np.float32)
  cdef float[:,::1] face_normal_view = face_normal
  cdef float * face_normal_buffer = &(face_normal_view[0,0])

  # select
  select_faces( \
    vertexBuffer, normalBuffer, \
    imgSizeV, \
    fM, T, \
    face_visible_buffer, \
    face_normal_buffer, \
    stats_buffer);

  return face_visible, face_normal, stats