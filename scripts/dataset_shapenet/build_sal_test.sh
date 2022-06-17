source dataset_shapenet/config.sh
# Make output directories
BUILD_PATH=$ROOT/data/ShapeNet.build_sal.test/
NPROC=0

mkdir -p $BUILD_PATH

# Run build
echo "Testing"
build_path_c=$BUILD_PATH

rm -r $build_path_c/4_points_direct \
      $build_path_c/4_sal
      
mkdir -p $build_path_c/0_in \
         $build_path_c/1_scaled \
         $build_path_c/1_transform \
         $build_path_c/2_depth \
         $build_path_c/4_pointcloud_direct \
         $build_path_c/4_sal
  
# echo "Scaling meshes"
# python $MESHFUSION_PATH/1_scale.py \
#   --n_proc $NPROC \
#   --in_dir $build_path_c/0_in \
#   --out_dir $build_path_c/1_scaled \
#   --t_dir $build_path_c/1_transform
  
# echo "Create depths maps"
# python $MESHFUSION_PATH/2_fusion.py \
#   --mode=render_new --n_proc $NPROC \
#   --in_dir $build_path_c/1_scaled \
#   --out_dir $build_path_c/2_depth \
#   --pointcloud_folder $build_path_c/4_pointcloud_direct \
#   --t_dir $build_path_c/1_transform \
#   --depth_offset_factor 0

echo "Produce watertight meshes"
python $MESHFUSION_PATH/2_fusion.py \
  --mode=judge_sal --n_proc $NPROC \
  --in_dir $build_path_c/4_pointcloud_direct_fps_N30000 \
  --out_dir $build_path_c/4_sal \
  --t_dir $build_path_c/1_transform \



