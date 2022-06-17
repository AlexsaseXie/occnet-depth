import os
import trimesh
import numpy as np
from tqdm import tqdm 
import shutil

SECTION3_GATHERED='/home2/xieyunwei/occupancy_networks/selected_models/selected_models/'
SECTION4_GATHERED='/home2/xieyunwei/occupancy_networks/selected_models_sal/selected_models_sal/'
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
SECTION3_OUTPUT_ROOT='./media/section3_gathered/'
SECTION4_OUTPUT_ROOT='./media/section4_gathered/'

def mkdir_p(a):
    if not os.path.exists(a):
        os.mkdir(a)

def copy_p(a, b):
    if not os.path.exists(b):
        shutil.copy(a, b)

def convert_off_to_obj(off_path, output_path, normalize=True):
    assert output_path.endswith('.obj')
    mesh = trimesh.load(off_path, process=False)
    if normalize:
        bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / 1.

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)
    mesh.export(output_path)
    
def convert_npz_pc_to_obj(npz_path, output_path, primitive='cube', resolution=2048):
    assert output_path.endswith('.obj')

    np_dict = np.load(npz_path)
    pc = np_dict['points']

    all_vertices = []
    all_faces = []
    rand_idx = np.random.choice(pc.shape[0], size=resolution, replace=False)
    for i, p in enumerate(tqdm(pc[rand_idx])):
        if primitive == 'sphere':
            tmp = trimesh.primitives.Sphere(radius=0.0025, center=p, subdivisions=1)
        elif primitive == 'cube':
            Rt = np.eye(4)
            Rt[:3,3] = p
            tmp = trimesh.primitives.Box(extents=(0.005,0.005,0.005), transform=Rt)  
        else:
            raise NotImplementedError

        current_len = len(tmp.vertices) * i

        all_vertices.append(np.array(tmp.vertices))
        all_faces.append(np.array(tmp.faces) + current_len)

    all_vertices = np.concatenate(all_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    mesh.export(output_path)

def regather_section3():
    mkdir_p(SECTION3_OUTPUT_ROOT)
    for c in CLASSES:
        class_root = os.path.join(SECTION3_GATHERED, c)
        output_root = os.path.join(SECTION3_OUTPUT_ROOT, c)
        if not os.path.exists(class_root):
            continue

        mkdir_p(output_root)

        modelnames = os.listdir(class_root)
        for modelname in tqdm(modelnames):
            model_root = os.path.join(class_root, modelname)

            gt_rgb = os.path.join(model_root, "gt_rgb.png")
            gt_mask = os.path.join(model_root, "gt_mask.png")
            gt_off = os.path.join(model_root, "gt.off")
            gt_camera = os.path.join(model_root, "cameras.npz")
            output_off = os.path.join(model_root, "ours.off")

            model_output_root = os.path.join(output_root, modelname)
            mkdir_p(model_output_root)

            copy_p(gt_rgb, os.path.join(model_output_root, "rgb.png"))
            copy_p(gt_mask, os.path.join(model_output_root, "mask.png"))
            convert_off_to_obj(gt_off, os.path.join(model_output_root, "gt.obj"))
            #copy_p(gt_off, os.path.join(model_output_root, "gt.off"))
            convert_off_to_obj(output_off, os.path.join(model_output_root, "ours.obj"))
            #copy_p(output_off, os.path.join(model_output_root, "ours.off"))
            copy_p(gt_camera, os.path.join(model_output_root, "cameras.npz"))

def regather_section4():
    mkdir_p(SECTION4_OUTPUT_ROOT)
    for c in CLASSES:
        class_root = os.path.join(SECTION4_GATHERED, c)
        output_root = os.path.join(SECTION4_OUTPUT_ROOT, c)
        if not os.path.exists(class_root):
            continue

        mkdir_p(output_root)

        modelnames = os.listdir(class_root)
        for modelname in tqdm(modelnames):
            model_root = os.path.join(class_root, modelname)

            gt_off = os.path.join(model_root, "gt.off")
            input_pc = os.path.join(model_root, "input_pc_30000.npz")
            sal_off = os.path.join(model_root, "sal.off")
            sail_s3_off = os.path.join(model_root, "sail_s3.off")
            input_img = os.path.join(model_root, "renderings", "gt.png")

            model_output_root = os.path.join(output_root, modelname)
            mkdir_p(model_output_root)

            copy_p(input_img, os.path.join(model_output_root, "gt_img.png"))
            convert_off_to_obj(gt_off, os.path.join(model_output_root, "gt.obj"))
            convert_off_to_obj(sal_off, os.path.join(model_output_root, "sal.obj"))
            convert_off_to_obj(sail_s3_off, os.path.join(model_output_root, "sail_s3.obj"))
            copy_p(input_pc, os.path.join(model_output_root, "input_pc_30000.npz"))
            convert_npz_pc_to_obj(input_pc, os.path.join(model_output_root, "input_pc_30000.obj"))

def test():
    model_root = os.path.join(SECTION3_GATHERED, '02691156', 'd441a12b217c26bc0d5f9d32d37453c')
    pc_model_root = os.path.join(SECTION4_GATHERED, '02691156', '103c9e43cdf6501c62b600da24e0965')
    test_output_root = '../data/web_gather_test/'
    mkdir_p(test_output_root)

    print('Begin:')
    convert_off_to_obj(os.path.join(model_root, 'gt.off'), os.path.join(test_output_root, 'gt_converted.obj'))
    print('Finish convert off to obj')
    #convert_npz_pc_to_obj(os.path.join(pc_model_root, 'input_pc_30000.npz'), os.path.join(test_output_root, 'pc_sphere.obj'))
    #print('Finish convert sphere pc to obj')
    convert_npz_pc_to_obj(os.path.join(pc_model_root, 'input_pc_30000.npz'), os.path.join(test_output_root, 'pc_cube.obj'), primitive='cube')
    print('Finish convert cube pc to obj')

if __name__ == '__main__':
    #test()

    regather_section3()
    regather_section4()