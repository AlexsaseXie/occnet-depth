source dataset_shapenet_with_depth/config.sh

# Function for processing a single model
reorganize_img_with_depth() {
  modelname=$(basename -- $5)
  output_path="$4/$modelname"
  build_path=$3
  vox_path=$2
  img_path=$1

  points_file="$build_path/4_points/$modelname.npz"
  points_out_file="$output_path/points.npz"

  pointcloud_file="$build_path/4_pointcloud/$modelname.npz"
  pointcloud_out_file="$output_path/pointcloud.npz"

  vox_file="$vox_path/$modelname/model.binvox"
  vox_out_file="$output_path/model.binvox"

  img_dir="$img_path/$modelname/rendering_png"
  img_out_dir="$output_path/img"
  depth_out_dir="$output_path/depth"

  if [! -f "$img_dir/depth_range.txt"]; then
    return
  fi

  metadata_file="$img_path/$modelname/rendering_metadata.txt"
  camera_out_file="$output_path/img/cameras.npz"

  echo "Copying model $output_path"
  mkdir -p $output_path $img_out_dir

  # points & pointcloud & voxel
  cp $points_file $points_out_file
  cp $pointcloud_file $pointcloud_out_file
  cp $vox_file $vox_out_file

  # camera
  python dataset_shapenet/get_r2n2_cameras.py $metadata_file $camera_out_file

  # copy regular png
  for f in $img_dir/*_rgb.png; do
    echo $f
    cp $f "$img_out_dir/"
  done

  # copy depth png
  for f in $img_dir/*_depth.png; do
    echo $f
    cp $f "$depth_out_dir/"
  done

  # copy depth range
  cp "$img_dir/depth_range.txt" "$depth_out_dir/"
}

export -f reorganize_img_with_depth

# Make output directories
mkdir -p $OUTPUT_PATH

# Run build
for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  BUILD_PATH_C=$BUILD_PATH/$c
  OUTPUT_PATH_C=$OUTPUT_PATH/$c
  IMG_PATH_C="$IMG_WITH_DEPTH_PATH/$c"
  VOX_PATH_C="$CHOY2016_PATH/ShapeNetVox32/$c"
  mkdir -p $OUTPUT_PATH_C

  ls $VOX_PATH_C | parallel -P $NPROC --timeout $TIMEOUT \
    reorganize_img_with_depth $IMG_PATH_C $VOX_PATH_C \
      $BUILD_PATH_C $OUTPUT_PATH_C {}
done
