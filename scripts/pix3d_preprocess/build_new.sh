source pix3d_preprocess/config.sh
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
           $build_path_c/4_points_direct_tsdf0 \
           $build_path_c/4_pointcloud_direct \

  echo "Converting meshes to OFF"
  lsfilter $input_path_c $build_path_c/0_in .off | parallel -P $NPROC --timeout $TIMEOUT \
     meshlabserver -i $input_path_c/{}/model.obj -o $build_path_c/0_in/{}.off -s pix3d_preprocess/rotate.mlx;
  
  echo "Scaling meshes"
  python $MESHFUSION_PATH/1_scale.py \
    --n_proc $NPROC \
    --in_dir $build_path_c/0_in \
    --out_dir $build_path_c/1_scaled \
    --t_dir $build_path_c/1_transform
  
  echo "Create depths maps"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=render_new --n_proc $NPROC \
    --in_dir $build_path_c/1_scaled \
    --out_dir $build_path_c/2_depth \
    --pointcloud_folder $build_path_c/4_pointcloud_direct \
    --t_dir $build_path_c/1_transform \
    --depth_offset_factor 0

  echo "Produce watertight meshes"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=judge_tsdf_view_pc --n_proc $NPROC \
    --in_dir $build_path_c/2_depth \
    --out_dir $build_path_c/4_points_direct_tsdf0 \
    --t_dir $build_path_c/1_transform \
    --pointcloud_folder $build_path_c/4_pointcloud_direct \
    --depth_offset_factor 0 \
    --truncation_factor 10 \
    --tsdf_offset 0.002 
done
