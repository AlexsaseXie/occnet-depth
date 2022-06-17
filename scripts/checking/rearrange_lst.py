import os
import shutil

PRE_BUILD_ROOT = '/home3/xieyunwei/ShapeNet.build.direct_remove/'
DATASET_PATH = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.update_lst/'
OUTPUT_LST_ROOT = '/home2/xieyunwei/occupancy_networks/data/ShapeNet.update_lst.remove.intersect/'

CLASSES = [
    '03001627',
    '02958343',
    '04256520',
    '02691156',
    '03636649',
    '04401088',
    '04530566',
    '03691459',
    '02933112',
    '04379243',
    '03211117',
    '02828884',
    '04090263',
]

SPLIT = ['train', 'val', 'test']

if not os.path.exists(OUTPUT_LST_ROOT):
    os.mkdir(OUTPUT_LST_ROOT)

for c in CLASSES:
    onet_class_root = os.path.join(DATASET_PATH, c)
    output_class_root = os.path.join(OUTPUT_LST_ROOT, c)

    if not os.path.exists(output_class_root):
        os.mkdir(output_class_root)
    else:
        shutil.rmtree(output_class_root)
        os.mkdir(output_class_root)

    info = {}

    for sp in SPLIT:
        onet_lst = os.path.join(onet_class_root, '%s.lst' % sp)
        with open(onet_lst, 'r') as f:
            onet_lst_modelnames = f.readlines()

        onet_lst_modelnames = list(filter(lambda x: len(x) > 5, onet_lst_modelnames))
        onet_lst_modelnames = list(map(lambda x: x.strip(), onet_lst_modelnames))

        output_lst = os.path.join(output_class_root, '%s.lst' % sp)

        drop_count = 0
        insert_list = []
        with open(output_lst, 'w') as f:
            for modelname in onet_lst_modelnames:
                check_path = os.path.join(PRE_BUILD_ROOT, c, '4_pointcloud_direct', '%s.npz' % modelname)

                if os.path.exists(check_path):
                    #f.write('%s\n' % modelname)
                    insert_list.append(modelname)
                else:
                    #print('%s doesn\'t exists' % check_path)
                    drop_count += 1

            f.write('\n'.join(insert_list))

        total_count = len(onet_lst_modelnames)
        remain_count = total_count - drop_count
        print('%s %s drop: %d/%d, remains: %d/%d' % (c, sp, drop_count, total_count, remain_count, total_count))

        info[sp] = drop_count / total_count

    print('Drop ratio train/val/test: %f/%f/%f' % (info['train'], info['val'], info['test']))