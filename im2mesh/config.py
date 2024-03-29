import yaml
from torchvision import transforms
from im2mesh import data
from im2mesh import onet, r2n2, psgn, pix2mesh, dmc
from im2mesh import onet_m
from im2mesh import onet_multi_layers_predict
from im2mesh import onet_depth
from im2mesh import point_completion
from im2mesh import faster_sail
from im2mesh import preprocess


method_dict = {
    'onet': onet,
    'r2n2': r2n2,
    'psgn': psgn,
    'pix2mesh': pix2mesh,
    'dmc': dmc,
    'onet_m': onet_m,
    'onet_multi_layers': onet_multi_layers_predict,
    'onet_depth': onet_depth,
    'point_completion': point_completion,
    'FCAE': point_completion,
    'MSN': point_completion,
    'SAL': faster_sail,
    'SAIL_S3': faster_sail
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    categories = cfg['data']['classes']

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.Shapes3dDataset(
            dataset_folder, fields,
            split=split,
            categories=categories,
        )
    elif dataset_type == 'Shapes3D_list':
        instance_list = cfg['data']['%s_instance_list' % mode] 
        fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        if return_category:
            fields['category'] = data.CategoryField()

        dataset = data.Shapes3dInstanceList_Dataset(
            dataset_folder, fields,
            train_list=instance_list,
        )
    elif dataset_type == 'kitti':
        dataset = data.KittiDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx
        )
    elif dataset_type == 'online_products':
        dataset = data.OnlineProductDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            classes=cfg['data']['classes'],
            max_number_imgs=cfg['generation']['max_number_imgs'],
            return_idx=return_idx, return_category=return_category
        )
    elif dataset_type == 'images':
        dataset = data.ImageDataset(
            dataset_folder, img_size=cfg['data']['img_size'],
            return_idx=return_idx,
        )
    elif dataset_type == 'pix3d':
        fields = {}

        fields = method_dict[method].config.get_data_fields(mode, cfg)

        if 'pix3d_root' in cfg['data']:
            pix3d_root = cfg['data']['pix3d_root']
        else:
            pix3d_root = '.'

        inputs_field = get_pix3d_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()
        
        dataset = data.Pix3dDataset(dataset_folder, fields, pix3d_root=pix3d_root)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])
 
    return dataset


def get_pix3d_inputs_field(mode, cfg):
    assert 'input_fields' in cfg['data']
    # only support mixed input field settings for pix3d dataset
    inputs_field_name = cfg['data']['input_fields']
    inputs_field = data.Pix3d_MixedInputField(inputs_field_name, mode, cfg)
    return inputs_field


def get_inputs_field(mode, cfg):
    ''' Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']
    if 'input_fields' in cfg['data']:
        # Mixed input field settings
        inputs_field_name = cfg['data']['input_fields']
        if inputs_field_name is None:
            return None
        inputs_field = data.MixedInputField(inputs_field_name, mode, cfg, n_views=24)
        return inputs_field
    with_transforms = cfg['data']['with_transforms']

    if input_type is None:
        inputs_field = None
    elif input_type == 'img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        if 'img_extension' in cfg['data']:
            inputs_field = data.ImagesField(
                cfg['data']['img_folder'], transform, 
                extension=cfg['data']['img_extension'],
                with_camera=with_camera, random_view=random_view
            )
        else:
            inputs_field = data.ImagesField(
                cfg['data']['img_folder'], transform,
                with_camera=with_camera, random_view=random_view
            )
    elif input_type == 'img_with_depth':
        # data augment not supported
        transform = transforms.Compose([
            transforms.Resize((cfg['data']['img_size'])), transforms.ToTensor(),
        ])
        
        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        data_params = {
            'with_camera': with_camera, 
            'random_view': random_view,
        }

        if 'absolute_depth' in cfg['data']:
            data_params['absolute_depth'] = cfg['data']['absolute_depth']

        if 'with_minmax' in cfg['data']:
            data_params['with_minmax'] = cfg['data']['with_minmax']

        if 'img_extension' in cfg['data']:
            data_params['extension'] = cfg['data']['img_extension']

        inputs_field = data.ImagesWithDepthField(
            'img', 'depth', 'mask', 
            transform, **data_params
        )
    elif input_type == 'depth_pred':
        # data augment not supported
        transform = transforms.Compose([
            transforms.Resize((cfg['data']['img_size'])), transforms.ToTensor(),
        ])
        
        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        data_params = {
            'with_camera': with_camera, 
            'random_view': random_view,
        }

        if 'absolute_depth' in cfg['data']:
            data_params['absolute_depth'] = cfg['data']['absolute_depth']

        if 'with_minmax' in cfg['data']:
            data_params['with_minmax'] = cfg['data']['with_minmax']

        if 'pred_with_img' in cfg['model']:
            data_params['with_img'] = cfg['model']['pred_with_img']

        if 'img_extension' in cfg['data']:
            data_params['extension'] = cfg['data']['img_extension']

        inputs_field = data.DepthPredictedField(
            'img', 'depth', 'mask', 
            cfg['data']['depth_pred_root'], 'depth_pred', transform, **data_params
        )
    elif input_type == 'depth_pointcloud' or input_type == 'depth_pointcloud_completion':
        t_lst = []
        if 'depth_pointcloud_n' in cfg['data'] and cfg['data']['depth_pointcloud_n'] is not None:
            t_lst.append(data.SubsamplePointcloud(cfg['data']['depth_pointcloud_n']))
        if 'depth_pointcloud_noise' in cfg['data'] and cfg['data']['depth_pointcloud_noise'] is not None:
            t_lst.append(data.PointcloudNoise(cfg['data']['depth_pointcloud_noise']))
        transform = transforms.Compose(t_lst)

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        data_params = {
            'random_view': random_view,
            'with_camera': True,
            'img_folder_name': 'img'
        }

        if 'view_penalty' in cfg['training'] and cfg['training']['view_penalty']:
            data_params['with_mask'] = True
            data_params['mask_folder_name'] = 'mask'
            data_params['mask_flow_folder_name'] = 'mask_flow'
            data_params['extension'] = 'png'
            img_transform = transforms.Compose([
                transforms.Resize((cfg['data']['img_size'])), transforms.ToTensor(),
            ])
            data_params['img_transform'] = img_transform
            data_params['with_depth_pred'] = True
            data_params['depth_pred_folder_name'] = 'depth_pred'
        
        inputs_field = data.DepthPointCloudField(
            cfg['data']['depth_pointcloud_root'],
            cfg['data']['depth_pointcloud_folder'],
            transform,
            **data_params
        )
    elif input_type == 'multi_img':
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])

        with_camera = cfg['data']['img_with_camera']

        if mode == 'train':
            random_view = True
        else:
            random_view = False

        inputs_field = data.ImagesField(
            cfg['data']['img_folder'], transform,
            with_camera=with_camera, random_view=random_view
        )
    elif input_type == 'pointcloud':
        transform = transforms.Compose([
            data.SubsamplePointcloud(cfg['data']['pointcloud_n']),
            data.PointcloudNoise(cfg['data']['pointcloud_noise'])
        ])
        with_transforms = cfg['data']['with_transforms']
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            with_transforms=with_transforms
        )
    elif input_type == 'voxels':
        inputs_field = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif input_type == 'idx':
        inputs_field = data.IndexField()
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field


def get_preprocessor(cfg, dataset=None, device=None):
    ''' Returns preprocessor instance.

    Args:
        cfg (dict): config dictionary
        dataset (dataset): dataset
        device (device): pytorch device
    '''
    p_type = cfg['preprocessor']['type']
    cfg_path = cfg['preprocessor']['config']
    model_file = cfg['preprocessor']['model_file']

    if p_type == 'psgn':
        preprocessor = preprocess.PSGNPreprocessor(
            cfg_path=cfg_path,
            pointcloud_n=cfg['data']['pointcloud_n'],
            dataset=dataset,
            device=device,
            model_file=model_file,
        )
    elif p_type is None:
        preprocessor = None
    else:
        raise ValueError('Invalid Preprocessor %s' % p_type)

    return preprocessor
