source dataset_shapenet/config.sh
# Make output directories
BUILD_PATH=$ROOT/data/ShapeNet.build.test/
NPROC=0

mkdir -p $BUILD_PATH

# Run build
echo "Testing"
build_path_c=$BUILD_PATH

rm -r $build_path_c/2_watertight \
      $build_path_c/4_points \
      $build_path_c/4_pointcloud \
      $build_path_c/4_points_direct \
      $build_path_c/2_depth \
      # $build_path_c/4_pointcloud_direct \
      # $build_path_c/4_watertight_scaled \
      
     


mkdir -p $build_path_c/0_in \
         $build_path_c/1_scaled \
         $build_path_c/1_transform \
         $build_path_c/2_depth \
         $build_path_c/2_watertight \
         $build_path_c/4_points \
         $build_path_c/4_pointcloud \
         $build_path_c/4_watertight_scaled \
         $build_path_c/4_pointcloud_direct \
         $build_path_c/4_points_direct
  
echo "Scaling meshes"
python $MESHFUSION_PATH/1_scale.py \
  --n_proc $NPROC \
  --in_dir $build_path_c/0_in \
  --out_dir $build_path_c/1_scaled \
  --t_dir $build_path_c/1_transform
  
# echo "Create depths maps"
# python $MESHFUSION_PATH/2_fusion.py \
#   --mode=render --n_proc $NPROC \
#   --in_dir $build_path_c/1_scaled \
#   --out_dir $build_path_c/2_depth \
#   --depth_offset_factor 0
# #  --n_views 500  


# echo "Produce watertight meshes"
# python $MESHFUSION_PATH/2_fusion.py \
#   --mode=fuse --n_proc $NPROC \
#   --in_dir $build_path_c/2_depth \
#   --out_dir $build_path_c/2_watertight \
#   --t_dir $build_path_c/1_transform \
#   --truncation_factor 10 \
#   --type tsdf
#   #--depth_offset_factor 0 \
#   #  --n_views 500

# echo "Produce direct points"
# python $MESHFUSION_PATH/2_fusion.py \
#   --mode=judge_inside_simple --n_proc $NPROC \
#   --in_dir $build_path_c/2_depth \
#   --out_dir $build_path_c/4_points_direct \
#   --t_dir $build_path_c/1_transform \
#   --packbits

# echo "Process watertight meshes"
# python sample_mesh.py $build_path_c/2_watertight \
#     --n_proc $NPROC --resize \
#     --bbox_in_folder $build_path_c/0_in \
#     --pointcloud_folder $build_path_c/4_pointcloud \
#     --points_folder $build_path_c/4_points \
#     --mesh_folder $build_path_c/4_watertight_scaled \
#     --packbits

# exit

echo "Create depths maps"
python $MESHFUSION_PATH/2_fusion.py \
  --mode=render_new --n_proc $NPROC \
  --in_dir $build_path_c/1_scaled \
  --out_dir $build_path_c/2_depth \
  --pointcloud_folder $build_path_c/4_pointcloud_direct \
  --t_dir $build_path_c/1_transform \
  --depth_offset_factor 0


# echo "Produce watertight meshes"
# python $MESHFUSION_PATH/2_fusion.py \
#   --mode=judge_tsdf_view_pc --n_proc $NPROC \
#   --in_dir $build_path_c/2_depth \
#   --out_dir $build_path_c/4_points_direct \
#   --t_dir $build_path_c/1_transform \
#   --pointcloud_folder $build_path_c/4_pointcloud_direct \
#   --depth_offset_factor 0 \
#   --truncation_factor 5 \
#   --packbits

# echo "Produce watertight meshes"
# python $MESHFUSION_PATH/2_fusion.py \
#   --mode=judge_tsdf_view_pc_according_to_pc --n_proc $NPROC \
#   --in_dir $build_path_c/2_depth \
#   --out_dir $build_path_c/4_points_direct \
#   --t_dir $build_path_c/1_transform \
#   --pointcloud_folder $build_path_c/4_pointcloud_direct \
#   --depth_offset_factor 0 \
#   --truncation_factor 10 \
#   --packbits



