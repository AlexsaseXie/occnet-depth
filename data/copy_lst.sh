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
A_PATH=/home2/xieyunwei/occupancy_networks/data/ShapeNet.update_lst.remove/
B_PATH=/home3/xieyunwei/ShapeNet.SAL/

for c in ${CLASSES[@]}; do
  echo "Processing $c"
  A_PATH_C=$A_PATH/$c
  B_PATH_C=$B_PATH/$c

  mkdir -p $B_PATH_C

  #mv $B_PATH_C/train.lst $B_PATH_C/train1.lst
  cp $A_PATH_C/train.lst $B_PATH_C/updated_train.lst

  #mv $B_PATH_C/val.lst $B_PATH_C/val1.lst
  #cp $A_PATH_C/val.lst $B_PATH_C/updated_val.lst

  #mv $B_PATH_C/test.lst $B_PATH_C/test1.lst
  #cp $A_PATH_C/test.lst $B_PATH_C/updated_test.lst
done


