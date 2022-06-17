# Code for XYW's master thesis

This repository contains the 3 main contributions of xyw's master thesis "3D reconstruction based on implicit function networks".
The code is constructed based on Occupancy Networks [Occupancy Networks - github](https://github.com/autonomousvision/occupancy_networks). For some of the issues you can refer to the original repository.

The 3 main contributions are:
1. A new ShapeNet preprocess algorithm which produces 3D ground truths with minimal shift over the raw ShapeNet models.
2. A new 3 stage single-view 3D reconstruction pipeline which achieves much higher IoU over the raw implicit surface network baseline ONet.
3. Code for SAL and RLIL point cloud surface reconstruction without normal input for challenging models on ShapeNet. The training uses unsigned loss.

## Installation
First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `mesh_funcspace` using
```
conda env create -f environment_new.yaml
```
[!Change]Check the environment.yaml if bug occurs. The main difference between the ONet's environment and ours is the pytorch version. We use 1.7+.
```
conda activate mesh_funcspace
```

Next, compile the extension modules.
You can do this via
```
python setup.py build_ext --inplace
python setup_c.py build_ext --inplace
```

[!Change] pytorch1.7+ has bug for the C module (especially pykdtree) so we use a separate install script.

## ONet's Original Dataset 

[!Change] We use ONet's original 3D data for training.

For ShapeNet dataset
To this end, there are two options:

1. you can download our preprocessed data
2. you can download the ShapeNet dataset and run the preprocessing pipeline yourself

Take in mind that running the preprocessing pipeline yourself requires a substantial amount time and space on your hard drive.
Unless you want to apply our method to a new dataset, we therefore recommmend to use the first option.

### Preprocessed data
You can download our preprocessed data (73.4 GB) using

```
bash scripts/download_data.sh
```

This script should download and unpack the data automatically into the `data/ShapeNet` folder.

### Building the dataset
Alternatively, you can also preprocess the dataset yourself.
To this end, you have to follow the following steps:
* download the [ShapeNet dataset v1](https://www.shapenet.org/) and put into `data/external/ShapeNet`. 
* download the [renderings and voxelizations](http://3d-r2n2.stanford.edu/) from Choy et al. 2016 and unpack them in `data/external/Choy2016` 
* build our modified version of [mesh-fusion](https://github.com/davidstutz/mesh-fusion) by following the instructions in the `external/mesh-fusion` folder

You are now ready to build the dataset:
```
cd scripts
bash dataset_shapenet/build.sh
``` 

This command will build the dataset in `data/ShapeNet.build`.
To install the dataset, run
```
bash dataset_shapenet/install.sh
```

If everything worked out, this will copy the dataset into `data/ShapeNet`.

## Additional Dataset
Our full dataset is built based on the original ONet's dataset.

The 2D view dataset of our paper is rebuilt due to the need of depth ground truth. The 3D dataset is also rebuilt in order to conduct fair and accurate evaluation compared to other methods.

### 2D view dataset with depth
The rendering script requires blender, install blender then
```
cd script/render_img_views/3D-R2N2/
CONFIG THE PATHS IN rendering_config.py THEN
python render_work.py
```

The rendering settings is the same with 3D-R2N2 but produces additional depth output. The output format for depth is a png gray image and a txt containing the min and max depth value for each depth map.

Then you can install a ShapeNet dataset with depth in which the folder structure is similar to the ONet's preprocess ShapeNet dataset.
```
cd script
CONFIG THE PATHS IN dataset_shapenet_with_depth/config.sh THEN
bash dataset_shapenet_with_depth/install
```

### A new 3D evaluation data *(Chapter 2)
The code for the new 3D ShapeNet preprocess data is in external/mesh-fusion/. Build the module first as the instruction by ONet.
The main contributed code is in librender/offscreen_new.cpp and libfusiongpu/fusion.cu. Then you can build the dataset by

```
cd script
CONFIG THE PATHS in dataset_shapenet/config.sh THEN
bash dataset_shapenet/build1_c1.sh
```

## 3 Stage Single-view 3D Reconstruction *(Chapter 3)
When you have installed all binary dependencies and obtained the preprocessed data, you are ready to run our pretrained models and train new models from scratch.

### (ShapeNet) Training
To train a new network from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory.

[!Change] To reproduce the results in Chapter 3, please use the configs in 
```
configs/img_depth_uniform/*
configs/img_depth_uniform_updated/* (FOR TESTING RESULTS IN PAPER)
```
### (ShapeNet)Generation
To generate meshes using a trained model, use
```
python generate_pred_depth_maps.py configs/img_depth_phase1/uresnet_origin_division.yaml --output_dir XXXXXX (Stage 1: Depth Estimation)
python generate_pointcloud_from_depth.py --mask_dir XXXXXX --output_dir XXXXXX (Stage 2: Back Projection)
python generate_pointcloud_completion.py configs/img_depth_uniform/* (Stage 2: Point Cloud Completion)
python generate.py CONFIG.yaml (Stage 3: Surface Reconstruction)
python generate_spsr.py (SPSR using meshlab)
```
where you replace `CONFIG.yaml` with the correct config file.

### (ShapeNet) Evaluation
For evaluation of the models, we provide two scripts: `eval.py` and `eval_meshes.py`.

The main evaluation script is `eval_meshes.py`.
You can run it using
```
python eval_meshes.py CONFIG.yaml
```
For a quick evaluation, you can also run
```
python eval.py CONFIG.yaml
```
All results reported in the paper were obtained using the `eval_meshes.py` script.

### (Pix3D) Evaluation
To test the ShapeNet pretrained model on Pix3D, first we need to generate input images and the ground truth 3D data.
```
cd scripts/pix3d_preprocess
bash run.sh
```

Then run by
```
python generate_*.py configs/pix3d/*.yaml
python eval*.py configs/pix3d/*.yaml
```

### Selecting the Images in the Paper
Please refer to the following folders
```
selected_models\
selected_models_pix3d\
selected_models_preprocess\
```

The selection first chooses the best models, then gathers different predictions by different methods into a single folder, then render images for each prediction and finally combine the images into a long sticker.

### Evaluating Foreign Methods
We provide scripts to eval AtlasNet, DISN and IM-Net as well.
```
python eval_xxxxx_meshsh.py --...
```

### Front End HTML Pages
A website based on django under the folder ShowResults/

Usage:
```
python organize_examples.py (GATHER REQUIRED RECONSTRUCTED MODELS)
python manage.py runserver 0.0.0.0:8001
```

You can browse the website in chrome by [127.0.0.1:8001](127.0.0.1:8001).

## Point Cloud Surface Reconstruction *(Chapter 4)

### SAL dataset
In order to train by unsigned loss, we need to produce unsigned ground truth samples. The point cloud we sampled the initial 100k point cloud into a 30k point cloud as the input.

First we create the list containing the first K models in each class of ShapeNet
```
cd scripts
python create_first_K_split.py
```

Then we sample each 100k point cloud on those K models.
```
CHANGE THE LIST OF OBJECTS THEN
python generate_pointcloud_sample.py configs/pointcloud_sample.yaml
```

We copy the models into the build folder by data/copy_points.lst. Finally we conduct unsigned ground truth building operation by
```
cd scripts
MODIFY THE PATHS IN dataset_shapenet/config.sh THEN
bash dataset_shapenet/build_sal_c.sh
MODIFY dataset_shapenet/install.sh THEN
bash dataset_shapenet/install
```

### SAL
Train by 
```
python sal_runner/train_single.py configs/sal/XXX.yaml
```
Generate by 
```
python sal_runner/generate_single.py configs/sal/XXX.yaml
```

Eval all meshes by 
```
CHANGE CONFIG IN sal_runner/eval_meshes.py THEN
python sal_runner/eval_meshes.py
```

### RLIL

Train by 
```
python sal_runner/train_sail.py configs/rlil/XXX.yaml
```
Generate by 
```
python sal_runner/generate_sail.py configs/rlil/XXX.yaml
python sal_runner/generate_sail_sep.py configs/rlil/XXX.yaml (FOR TESTING ON INDIVIDUAL LOCAL CUBES)
```

Eval all meshes by 
```
CHANGE CONFIG IN sal_runner/eval_meshes.py THEN
python sal_runner/eval_meshes.py
```

## Contact Author

In case you have other questions about the code, you can contact me by email.

Email: thss15_xieyw@163.com
Xie Yunwei