declare -a CLASSES=(
03001627
02958343
04256520
02691156
03636649
04401088
04530566
03691459
02933112
04379243
03211117
02828884
04090263
)

ROOT=.
A_PATH=$ROOT/ShapeNet.with_depth.10w10w
#B_PATH=/home1/xieyunwei/ShapeNet.with_depth.mask_flow

# mkdir -p $B_PATH
for c in ${CLASSES[@]}; do
  echo "Processing $c"
  A_PATH_C=$A_PATH/$c
  B_PATH_C=$B_PATH/$c
  #mkdir -p $B_PATH_C

  for model in `ls $A_PATH_C`; do
    if [ -d $A_PATH_C/$model ]; then
      #mkdir -p "$B_PATH_C/$model"
      rm "$A_PATH_C/$model/points_direct.npz"
      rm "$A_PATH_C/$model/points_direct_new.npz"

      # if [ -f "$A_PATH_C/$model/points_transfered.npz" ]; then
      #   rm "$A_PATH_C/$model/points_transfered.npz"
      #   rm "$A_PATH_C/$model/points_uniform500000.h5"
      #   #rm "$A_PATH_C/$model/points_sdf50000.npz"

      #   echo "finished $c/$model"
      # fi

      #cp "$A_PATH_C/$model/points_uniform_sdf150000.h5" "$B_PATH_C/$model/"
      #cp "$A_PATH_C/$model/pointcloud.npz" "$B_PATH_C/$model/"
      #cp "$A_PATH_C/$model/model.binvox" "$B_PATH_C/$model/" 

      #cp -r "$A_PATH_C/$model/depth" "$B_PATH_C/$model/"
      #cp -r "$A_PATH_C/$model/depth_pointcloud" "$B_PATH_C/$model/"
      #cp -r "$A_PATH_C/$model/mask" "$B_PATH_C/$model/"
      #cp -r "$A_PATH_C/$model/img" "$B_PATH_C/$model/"
      echo "finished $c/$model"
    fi
  done

  #zip -q -r "$B_PATH/$c.zip" "$B_PATH_C/"  
done
