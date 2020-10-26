import os
import sys
from rendering_config import *

def main():
    print('Rendering')
    for model_class in CLASSES:
        missing_count = 0

        class_root = os.path.join(SHAPENET_ROOT, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_root)

        for model_id in current_class_ids:
            rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, model_class, model_id)
            if not os.path.exists(os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % (N_VIEWS - 1))):
                #print('%s/%s is missing' % (model_class, model_id))
                missing_count += 1

        print('Class %s missing: %d / %d' % (model_class, missing_count, len(current_class_ids)))

    print('Transfer')
    for model_class in CLASSES:
        missing_count = 0

        class_root = os.path.join(SHAPENET_ROOT, model_class)
        #class_root = os.path.join(R2N2_ROOT, model_class)
        current_class_ids = os.listdir(class_root)

        for model_id in current_class_ids:
            rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, model_class, model_id)
            if not os.path.exists(os.path.join(rendering_curr_model_root, 'rendering_png', '%.2d_rgb.png' % (N_VIEWS - 1))): 
                #print('%s/%s is missing' % (model_class, model_id))
                missing_count += 1

        print('Class %s missing: %d / %d' % (model_class, missing_count, len(current_class_ids)))

if __name__ == '__main__':
    main()
