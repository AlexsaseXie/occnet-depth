import numpy as np
import os
import glob
import sys
import random
from tqdm import tqdm

model_cls = ['03001627','02958343','04256520','02691156','03636649','04401088','04530566','03691459','02933112','04379243','03211117','02828884','04090263']

ROOT = 'data/ShapeNet.with_depth.10w10w'
POINTS_COUNT = [8192, 4096, 2048, 1024, 512, 256, 128]
RADIUS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4]
MODEL_TEST_COUNT = 100

class_average = np.zeros((len(model_cls), len(POINTS_COUNT), len(RADIUS)))
for c_i,c in enumerate(model_cls):
    print('processing class:', c)
    class_root = os.path.join(ROOT, c)
    input_dirs = os.listdir(class_root)
    input_dirs = list(filter(lambda x: not x.endswith('.lst'), input_dirs))

    model_count = len(input_dirs)
    chosen_idx = np.random.choice(model_count, MODEL_TEST_COUNT, replace=False)

    model_average = np.zeros((MODEL_TEST_COUNT, len(POINTS_COUNT), len(RADIUS)))
    for model_i, model_idx in enumerate(chosen_idx):
        model_root = os.path.join(class_root, input_dirs[model_idx])

        pc_file = os.path.join(model_root, 'pointcloud.npz')
        raw_pc_npz = np.load(pc_file)
        raw_pc = raw_pc_npz['points']

        for pc_i,pc_size in enumerate(POINTS_COUNT):
            pointcloud = np.random.choice(raw_pc.shape[0], pc_size, replace=False)
            pointcloud = raw_pc[pointcloud]

            test_case = pc_size // 20

            total_num = np.zeros((len(RADIUS), test_case))
            for i in range(test_case):
                k = np.random.randint(pc_size)
                vec = pointcloud[k]

                distance = np.sum((pointcloud - vec) ** 2, axis=1)
                for idx, r in enumerate(RADIUS):
                    num = (distance <= r * r).sum()
                    total_num[idx, i] = num
                
            mean_num = total_num.mean(axis=1)
            max_num = total_num.max(axis=1)
            min_num = total_num.min(axis=1)

            for r_idx, r in enumerate(RADIUS):
                model_average[model_i, pc_i, r_idx] = mean_num[r_idx]

                #print('%s/%s:' % (c, input_dirs[model_idx]),'Average %.2f neighbor count of %d pc is' % (r,pc_size), mean_num[r_idx], 'range from [%f, %f]' % (min_num[r_idx], max_num[r_idx]))


    cur_class_average = model_average.mean(axis=0)

    class_average[c_i, :, :] = cur_class_average
    print('---CLASS %s' % c)
    for pc_i, pc_size in enumerate(POINTS_COUNT):
        for r_i, r in enumerate(RADIUS):
            print('Average %.2f neighbor count of  %d pc is %.2f' % (r, pc_size, cur_class_average[pc_i, r_i]) )

all_average = class_average.mean(axis=0)
print('---ALL AVERAGE')
for pc_i, pc_size in enumerate(POINTS_COUNT):
    for r_i, r in enumerate(RADIUS):
        print('Average %.2f neighbor count of  %d pc is %.2f' % (r, pc_size, all_average[pc_i, r_i]) )
