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
  rm -r $build_path_c/4_sal_30000_copy 
        #$build_path_c/4_sal_50000

  mkdir -p $build_path_c/4_sal_30000_copy 
           #$build_path_c/4_sal_50000 

  # echo "Create depths maps"
  # python $MESHFUSION_PATH/2_fusion.py \
  #   --mode=render_new --n_proc $NPROC \
  #   --in_dir $pre_build_path_c/1_scaled \
  #   --out_dir $build_path_c/2_depth \
  #   --pointcloud_folder $build_path_c/4_pointcloud_direct \
  #   --t_dir $pre_build_path_c/1_transform \
  #   --depth_offset_factor 0

  echo "SAL 30000"
  python $MESHFUSION_PATH/2_fusion.py \
    --mode=judge_sal --n_proc $NPROC \
    --in_dir $build_path_c/4_pointcloud_direct_fps_N30000_copy \
    --out_dir $build_path_c/4_sal_30000_copy \
    --t_dir $pre_build_path_c/1_transform 

  # echo "SAL 50000"
  # python $MESHFUSION_PATH/2_fusion.py \
  #   --mode=judge_sal --n_proc $NPROC \
  #   --in_dir $build_path_c/4_pointcloud_direct_fps_N50000 \
  #   --out_dir $build_path_c/4_sal_50000 \
  #   --t_dir $pre_build_path_c/1_transform 
  
done
