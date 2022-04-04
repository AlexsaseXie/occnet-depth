try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
import os
import glob


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'im2mesh/utils/libmcubes/mcubes.pyx',
        'im2mesh/utils/libmcubes/pywrapper.cpp',
        'im2mesh/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# DMC extensions
dmc_pred2mesh_module = CppExtension(
    'im2mesh.dmc.ops.cpp_modules.pred2mesh',
    sources=[
        'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
    ]   
)

dmc_cuda_module = CUDAExtension(
    'im2mesh.dmc.ops._cuda_ext', 
    sources=[
        'im2mesh/dmc/ops/src/extension.cpp',
        'im2mesh/dmc/ops/src/curvature_constraint_kernel.cu',
        'im2mesh/dmc/ops/src/grid_pooling_kernel.cu',
        'im2mesh/dmc/ops/src/occupancy_to_topology_kernel.cu',
        'im2mesh/dmc/ops/src/occupancy_connectivity_kernel.cu',
        'im2mesh/dmc/ops/src/point_triangle_distance_kernel.cu',
    ]
)


# added to compile pointnet2_op_lib
this_dir = os.path.dirname(os.path.abspath(__file__))
_ext_src_root = os.path.join("im2mesh","utils", "pointnet2_ops_lib","pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(os.path.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    os.path.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(os.path.join(this_dir, _ext_src_root, "include", "*"))


pointnet2_cuda_module = CUDAExtension(
    name="im2mesh.utils.pointnet2_ops_lib.pointnet2_ops._ext",
    sources=_ext_sources,
    extra_compile_args={
        "cxx": ["-O3", '-std=c++14'],
        "nvcc": ["-O3"],
    },
    include_dirs=[os.path.join(this_dir, _ext_src_root, "include")],
)

libpcd_src_root = os.path.join("im2mesh","utils", "lib_pointcloud_distance", )
libpcd_ext_sources = glob.glob(os.path.join(libpcd_src_root, "src", "*.cpp")) + glob.glob(
    os.path.join(libpcd_src_root, "src", "*.cu")
)


lib_pointcloud_distance_module = CUDAExtension(
    name="im2mesh.utils.lib_pointcloud_distance._ext",
    sources=libpcd_ext_sources,
    extra_compile_args={
        "cxx": ["-O3", '-std=c++14'],
        "nvcc": ["-O3"],
    },
    include_dirs=[os.path.join(this_dir, libpcd_src_root, "include")],
)

MSN_utils_src_root = os.path.join("im2mesh","point_completion", "MSN_utils", )
MSN_utils_ext_sources = glob.glob(os.path.join(MSN_utils_src_root, "src", "*.cpp")) + glob.glob(
    os.path.join(MSN_utils_src_root, "src", "*.cu")
)


MSN_module = CUDAExtension(
    name="im2mesh.point_completion.MSN_utils._ext",
    sources=MSN_utils_ext_sources,
    extra_compile_args={
        "cxx": ["-O3", '-std=c++14'],
        "nvcc": ["-O3"],
    },
    include_dirs=[os.path.join(this_dir, MSN_utils_src_root, "include")],
)

Space_carver_sampler_module_root = os.path.join("im2mesh", "onet_depth", "models", "space_carver_ops")
Space_carver_sampler_module_sources = glob.glob(os.path.join(Space_carver_sampler_module_root, "src", "*.cpp")) + glob.glob(
    os.path.join(Space_carver_sampler_module_root, "src", "*.cu"))

Space_carver_sampler_module = CUDAExtension(
    name="im2mesh.onet_depth.models.space_carver_ops._ext",
    sources=Space_carver_sampler_module_sources,
    extra_compile_args={
        "cxx": ["-O3", '-std=c++14'],
        "nvcc": ["-O3"],
    }
)

pointcloud_voxel_module = Extension(
    'im2mesh.utils.lib_pointcloud_voxel.pointcloud_voxel',
    sources=[
        'im2mesh/utils/lib_pointcloud_voxel/pointcloud_voxel.pyx',
        'im2mesh/utils/lib_pointcloud_voxel/pointcloud_voxel_c.cpp',
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# Gather all extension modules
ext_modules = [
    #pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    dmc_pred2mesh_module,
    dmc_cuda_module,
    pointnet2_cuda_module,
    lib_pointcloud_distance_module,
    MSN_module,
    Space_carver_sampler_module,
    pointcloud_voxel_module
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmdclass={
        'build_ext': BuildExtension
    }
)
