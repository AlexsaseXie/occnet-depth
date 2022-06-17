import os
import trimesh
from tqdm import tqdm

ROOT = '/home2/xieyunwei/occupancy_networks/out/img_depth_uniform/phase2_depth_pointcloud_MSN_space_carved_mixed_local/generation_space_carved/'

test_root = os.path.join(ROOT, 'meshes')

#test_root = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.build/'

for classname in os.listdir(test_root):
    if not os.path.isdir(os.path.join(test_root, classname)):
        continue
    not_watertight_count = 0
    total_count = 0
    class_root = os.path.join(test_root, classname)
    
    print('Class %s' % classname)
    for model_name in tqdm(os.listdir(class_root)):
        model_path = os.path.join(class_root, model_name)

        mesh = trimesh.load(model_path, process=False)

        total_count += 1
        if not mesh.is_watertight:
            not_watertight_count += 1
            #print('Not watertight')

    print("%s has %d/%d none watertight meshes" % (classname, not_watertight_count, total_count)) 


