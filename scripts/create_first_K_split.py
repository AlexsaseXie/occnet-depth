import argparse
import os
import glob
from unicodedata import category


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home3/xieyunwei/ShapeNet.SAL/')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--previous_list', type=str, default='updated_train.lst')
parser.add_argument('--output_list', type=str, default='train_instance.lst')

args = parser.parse_args()

categories = os.listdir(args.dataset_path)
for c in categories:
    category_path = os.path.join(args.dataset_path, c)
    if not os.path.isdir(category_path):
        continue

    lst_path = os.path.join(category_path, args.previous_list)
    
    assert os.path.exists(lst_path)
    with open(lst_path, 'r') as f:
        models_c = f.read().split('\n')
    
    output_path = os.path.join(category_path, args.output_list)
    with open(output_path, 'w') as f:
        for i in range(args.K):
            f.write('%s %s' % (c, models_c[i]))
            if i != args.K - 1:
                f.write('\n')
    

    


    

