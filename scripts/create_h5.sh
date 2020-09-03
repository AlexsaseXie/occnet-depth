source dataset_shapenet/config.sh

TRANSFER_PATH=$ROOT/data/ShapeNet.with_depth.10w10w

for c in ${CLASSES[@]}; do
  echo "Parsing class $c"
  TRANSFER_PATH_C=$TRANSFER_PATH/$c
  CHOY2016_VOX_PATH_C="$CHOY2016_PATH/ShapeNetVox32/$c"

  ls $CHOY2016_VOX_PATH_C | parallel -P $NPROC --timeout $TIMEOUT \
    python create_h5.py "$TRANSFER_PATH_C/{}" --unpack_bits

done
