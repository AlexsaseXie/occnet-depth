from torch import nn
from torchvision import transforms
from im2mesh import config, data, common

IMG_SIZE = 224

def get_img_inputs_field(mode):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE)), 
        transforms.ToTensor(),
    ])

    with_camera = False

    if mode == 'train':
        random_view = True
    else:
        random_view = False

    inputs_field = data.ImagesField(
        'img_choy2016', transform,
        with_camera=with_camera, random_view=random_view
    )

    return inputs_field

def get_img_with_depth_input_field(mode):
    # data augment not supported
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE)), 
        transforms.ToTensor(),
    ])
    
    with_camera = False

    if mode == 'train':
        random_view = True
    else:
        random_view = False

    inputs_field = data.ImagesWithDepthField(
        'img', 'depth', 'mask', transform,
        with_camera=with_camera, random_view=random_view
    )
    return inputs_field

def get_dataset(dataset_root, mode, input_type='img'):
    if input_type == 'img':
        inputs_field = get_img_inputs_field(mode)
    elif input_type == 'img_with_depth':
        inputs_field = get_img_with_depth_input_field(mode)

    fields = {}

    fields['inputs'] = inputs_field
    fields['idx'] = data.IndexField()
    fields['category'] = data.CategoryField()

    dataset = data.Shapes3dDataset(
        dataset_root, fields,
        split=mode,
        categories=None,
    )
    return dataset
 