# TODO:
# make following finetune methods work
# * need depth ground truth

# depth prediction train
python train.py configs/pix3d/uresnet_finetune.yaml

# depth prediction
python pix3d_test/generate_pred_depth_maps.py configs/pix3d/uresnet_finetune.yaml --out_dir ./data/pix3d/uresnet.depth_pred/

# inverse mapping of perspective projection
python pix3d_test/generate_pointcloud_from_depth.py

# MSN point cloud completion
python pix3d_test/generate_pointcloud_completion.py configs/pix3d/MSN.yaml --out_folder_name MSN_mixed_space_carved_4096 --resample 4096

# onet generation
python pix3d_test/generate.py configs/pix3d/phase2.yaml

# eval mesh
python pix3d_test/eval_meshes.py configs/pix3d/phase2.yaml

# eval
python pix3d_test/eval.py configs/pix3d/phase2.yaml