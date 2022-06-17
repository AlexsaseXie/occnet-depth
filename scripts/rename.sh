source dataset_shapenet/config.sh

RENAME_PATH=$ROOT/data/ShapeNet.with_depth.10w10w
A_NAME=points_uniform500000.npz
B_NAME=points_surface_0.5.npz

for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  RENAME_PATH_C=$RENAME_PATH/$c
  CHOY2016_VOX_PATH_C="$CHOY2016_PATH/ShapeNetVox32/$c"

  for dir in `ls $CHOY2016_VOX_PATH_C`
  do
    rm "$RENAME_PATH_C/$dir/$A_NAME"
    #mv "$RENAME_PATH_C/$dir/$A_NAME" "$RENAME_PATH_C/$dir/$B_NAME"
  done
done
