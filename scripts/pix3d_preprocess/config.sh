ROOT=..

export MESHFUSION_PATH=$ROOT/external/mesh-fusion
export HDF5_USE_FILE_LOCKING=FALSE # Workaround for NFS mounts

INPUT_PATH=/home2/xieyunwei/pix3d/model/
BUILD_PATH=$ROOT/data/pix3d/pix3d.build

NPROC=12
TIMEOUT=180

declare -a CLASSES=(
    sofa
    chair
    table
)

# Utility functions
lsfilter() {
 folder=$1
 other_folder=$2
 ext=$3

 for f in $folder/*; do
   filename=$(basename $f)
   if [ ! -f $other_folder/$filename$ext ] && [ ! -d $other_folder/$filename$ext ]; then
    echo $filename
   fi
 done
}
