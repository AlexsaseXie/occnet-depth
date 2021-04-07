from torch import nn
from torchvision import transforms
from im2mesh import config, data, common

IMG_SIZE = 224

def get_img_inputs_field(mode, cfg):
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
        cfg['data']['img_folder'], transform, extension=cfg['data']['img_extension'],
        with_camera=with_camera, random_view=random_view
    )

    return inputs_field

def get_img_with_depth_input_field(mode, cfg):
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

    if 'absolute_depth' in cfg['data']:
        absolute_depth = cfg['data']['absolute_depth']
    else:
        absolute_depth = False

    inputs_field = data.ImagesWithDepthField(
        'img', 'depth', 'mask', transform,
        with_camera=with_camera, random_view=random_view,
        absolute_depth=absolute_depth
    )
    return inputs_field

def get_depth_pointcloud_field(mode, cfg):
    if mode == 'train':
        random_view = True
    else:
        random_view = False

    t_lst = []
    transform = transforms.Compose(t_lst)
    inputs_field = data.DepthPointCloudField(
        cfg['data']['depth_pointcloud_root'],
        cfg['data']['depth_pointcloud_folder'],
        transform,
        random_view=random_view,
        with_camera=True,
        img_folder_name='img'
    )
    return inputs_field

def get_dataset(dataset_root, mode, cfg, input_type='img'):
    if 'input_fields' in cfg['data']:
        # Mixed input field settings
        inputs_field_name = cfg['data']['input_fields']
        inputs_field = data.MixedInputField(inputs_field_name, mode, cfg, n_views=24)
    else:
        if input_type == 'img':
            inputs_field = get_img_inputs_field(mode, cfg)
        elif input_type == 'img_with_depth':
            inputs_field = get_img_with_depth_input_field(mode, cfg)
        elif input_type in ('depth_pointcloud', 'depth_pointcloud_completion'):
            inputs_field = get_depth_pointcloud_field(mode, cfg)

    fields = {}

    fields['inputs'] = inputs_field
    fields['idx'] = data.IndexField()
    fields['category'] = data.CategoryField()

    if input_type in ('depth_pointcloud', 'depth_pointcloud_completion') and cfg['model']['depth_pointcloud_transfer'] == 'world_normalized':
        # only points loc & scale are needed
        fields['points'] = data.PointsField(
            'points.npz', transform=None,
            with_transforms=True, unpackbits=True
        )

    dataset = data.Shapes3dDataset(
        dataset_root, fields,
        split=mode,
        categories=None,
    )
    return dataset
 
