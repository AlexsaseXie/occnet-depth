# depth prediction
python pix3d_test/generate_pred_depth_maps.py configs/pix3d/uresnet.yaml

# inverse mapping of perspective projection
python pix3d_test/generate_pointcloud_from_depth.py

# MSN point cloud completion
python pix3d_test/generate_pointcloud_completion.py configs/pix3d/MSN.yaml --out_folder_name MSN_mixed_space_carved_4096 --resample 4096

