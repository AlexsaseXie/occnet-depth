ROOT=..

export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=$ROOT/external/ShapeNetCore.v1
CHOY2016_PATH=$ROOT/external/Choy2016
BUILD_PATH=$ROOT/data/ShapeNet.testbuild
OUTPUT_PATH=$ROOT/data/ShapeNet

NPROC=12
TIMEOUT=180
N_VAL=100
N_TEST=100
N_AUG=50

declare -a CLASSES=(
03001627
)


# Make output directories
mkdir -p $BUILD_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c

  mkdir -p $build_path_c/0_in \
           $build_path_c/1_scaled \
           $build_path_c/1_transform \
           $build_path_c/2_depth \
           $build_path_c/2_watertight \
           $build_path_c/4_points \
           $build_path_c/4_pointcloud \
           $build_path_c/4_watertight_scaled \

  echo "Converting meshes to OFF"
  meshlabserver -i $input_path_c/7f271ecbdeb7610d637adadafee6f182/model.obj -o $build_path_c/0_in/7f271ecbdeb7610d637adadafee6f182.off;
  
  echo "Scaling meshes"
  python $MESHFUSION_PATH/1_scale.py \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform
  
  echo "Create depths maps"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=render \
    --in_dir $build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth
  
  echo "Produce watertight meshes"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=fuse \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/2_watertight \
    --t_dir $build_path_c/1_transform

  echo "Process watertight meshes"
  python sample_mesh1.py $build_path_c/2_watertight \
      --resize \
      --points_size 100000 \
      --points_uniform_ratio 0.7 \
      --bbox_in_folder $build_path_c/0_in \
      --pointcloud_folder $build_path_c/4_pointcloud \
      --points_folder $build_path_c/4_points \
      --mesh_folder $build_path_c/4_watertight_scaled \
      --packbits --float16
done
