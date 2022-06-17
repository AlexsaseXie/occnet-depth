source dataset_shapenet/config.sh
# Make output directories
mkdir -p $BUILD_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Processing class $c"
  input_path_c=$INPUT_PATH/$c
  build_path_c=$BUILD_PATH/$c
  pre_build_path_c=$PRE_BUILD_PATH/$c
  
  echo "Pre build_path: $pre_build_path_c"
  mkdir -p $build_path_c/4_points_direct_tsdf0.008 \
           $build_path_c/4_pointcloud_direct \
           $build_path_c/2_depth

  echo "Create depths maps"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=render_new --n_proc $NPROC \
    --in_dir $pre_build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth \
    --pointcloud_folder $build_path_c/4_pointcloud_direct \
    --t_dir $pre_build_path_c/1_transform \
    --depth_offset_factor 0


  echo "Produce watertight meshes"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=judge_tsdf_view_pc --n_proc $NPROC \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/4_points_direct_tsdf0.008 \
    --t_dir $pre_build_path_c/1_transform \
    --pointcloud_folder $build_path_c/4_pointcloud_direct \
    --depth_offset_factor 0 \
    --truncation_factor 10 \
    --packbits

done
