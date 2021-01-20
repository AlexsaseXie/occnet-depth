mkdir -p /home2/xieyunwei/occupancy_networks/data/pix3d

echo 'Sampling'
python 0_sample.py --pix3d_root /home2/xieyunwei/pix3d/ --sampled_list_root /home2/xieyunwei/occupancy_networks/data/pix3d/sampled_list/

echo 'Generating'
python 1_generate_test_imgs.py --pix3d_root /home2/xieyunwei/pix3d/ --sampled_list_root /home2/xieyunwei/occupancy_networks/data/pix3d/sampled_list/ --output_list_root /home2/xieyunwei/occupancy_networks/data/pix3d/generation_sampled_list/ --generate_root /home2/xieyunwei/occupancy_networks/data/pix3d/generation/ --fit_focal_length --intermediate_output --test_num -1

