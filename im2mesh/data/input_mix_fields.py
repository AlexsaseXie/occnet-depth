import os
import glob
import random
import numpy as np
from im2mesh.data.core import Field
import torch
from torchvision import transforms
from im2mesh.data.fields import get_depth_image, get_mask, get_image, get_mask_flow
from im2mesh.data.fields import IndexField, ViewIdField, CategoryField
from im2mesh.data.transforms import SubsamplePointcloud, PointcloudNoise, ShufflePointcloud, PointcloudDropout

class S_ImagesField(Field):
    ''' S_Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
    '''
    def __init__(self, folder_name, transform=None, extension='jpg'):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        assert view_id is not None
        idx_img = view_id
        
        image = get_image(folder, idx_img, transform=self.transform, extension=self.extension)

        data = {
            None: image
        }

        return data

class S_CameraField(Field):
    ''' S_Camera Field.

    It is the field used for loading camera args.

    Args:
        folder_name (str): folder name
    '''
    def __init__(self, folder_name):
        self.folder_name = folder_name

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        assert view_id is not None
        idx_img = view_id

        data = {}

        camera_file = os.path.join(folder, 'cameras.npz')
        camera_dict = np.load(camera_file)
        Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
        K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
        data['world_mat'] = Rt
        data['camera_mat'] = K

        return data

class S_DepthField(Field):
    ''' S_Depth Field.

    It is the field used for loading depth.

    Args:
        depth_folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        absolute_depth (bool): whether return absolute depth or not
        with_minmax: whether return depth min & max or not
    '''
    def __init__(self, folder_name='depth', transform=None,
                 extension='png', absolute_depth=True, with_minmax=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.absolute_depth = absolute_depth
        self.with_minmax = with_minmax

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        depth_folder = os.path.join(model_path, self.folder_name)

        assert view_id is not None
        idx_img = view_id

        depth_image, depth_min, depth_max = get_depth_image(depth_folder, idx_img, transform=self.transform, extension=self.extension)
        if self.absolute_depth:
            depth_image = depth_image * (depth_max - depth_min) + depth_min

        data = {
            None: depth_image,
        }

        if self.with_minmax:
            data['depth_min'] = depth_min
            data['depth_max'] = depth_max

        return data

class S_MaskField(Field):
    ''' S_Mask Field.

    It is the field used for loading mask.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        absolute_depth (bool): whether return absolute depth or not
        with_minmax: whether return depth min & max or not
    '''
    def __init__(self, folder_name='mask', transform=None, extension='png'):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        mask_folder = os.path.join(model_path, self.folder_name)

        assert view_id is not None
        idx_img = view_id

        depth_mask = get_mask(mask_folder, idx_img, transform=self.transform, extension=self.extension)

        data = {
            None: depth_mask
        }

        return data

class S_MaskFlowField(Field):
    ''' Mask Flow Field.

    It is the field used for loading mask flow.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
    '''
    def __init__(self, mask_flow_folder_name='mask_flow',
                  transform=None, extension='png'):
        self.mask_flow_folder_name = mask_flow_folder_name
        self.extension = extension
        self.transform = transform

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        assert view_id is not None
        idx_img = view_id

        mask_flow_folder = os.path.join(model_path, self.mask_flow_folder_name)
        mask_flow = get_mask_flow(mask_flow_folder, idx_img, transform=self.transform, extension=self.extension)
        
        data = {
            None: mask_flow
        }

        return data

class S_DepthPredField(Field):
    ''' S_DepthPred Field.

    It is the field used for loading predicted depth maps.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
    '''
    def __init__(self, depth_pred_root=None, depth_pred_folder_name='depth_pred',
                  transform=None, extension='png', absolute_depth=True, with_minmax=False):
        self.depth_pred_root = depth_pred_root
        self.depth_pred_folder_name = depth_pred_folder_name
        self.transform = transform
        self.extension = extension
        self.absolute_depth = absolute_depth
        self.with_minmax = with_minmax

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        assert view_id is not None
        idx_img = view_id

        if self.depth_pred_root is not None:
            paths = model_path.split('/')
            depth_pred_folder = os.path.join(self.depth_pred_root, 
                paths[-2], paths[-1], 
                self.depth_pred_folder_name)
        else:
            depth_pred_folder = os.path.join(model_path, self.depth_pred_folder_name)

        depth_pred_image, depth_pred_min, depth_pred_max = get_depth_image(depth_pred_folder, idx_img, transform=self.transform, extension=self.extension)
            
        if self.absolute_depth:
            depth_pred_image = depth_pred_image * (depth_pred_max - depth_pred_min) + depth_pred_min

        data = {
            None: depth_pred_image
        }

        if self.with_minmax:
            data['depth_pred_min'] = depth_pred_min
            data['depth_pred_max'] = depth_pred_max

        return data

class S_DepthPointCloudField(Field):
    ''' Depth Point Cloud Field.

    It is the field used for loading depth point cloud.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
    '''
    def __init__(self, depth_pointcloud_root=None, depth_pointcloud_folder_name='depth_pointcloud',
                  transform=None, mixed=False):
        self.depth_pointcloud_root = depth_pointcloud_root
        self.depth_pointcloud_folder_name = depth_pointcloud_folder_name
        self.transform = transform
        self.mixed = mixed

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if not self.mixed:
            if self.depth_pointcloud_root is not None:
                paths = model_path.split('/')
                depth_pointcloud_folder = os.path.join(self.depth_pointcloud_root, paths[-2], paths[-1], self.depth_pointcloud_folder_name)
            else:
                depth_pointcloud_folder = os.path.join(model_path, self.depth_pointcloud_folder_name)
        else:
            assert self.depth_pointcloud_root is not None
            if random.random() >= 0.5:
                paths = model_path.split('/')
                depth_pointcloud_folder = os.path.join(self.depth_pointcloud_root, paths[-2], paths[-1], self.depth_pointcloud_folder_name)
            else:
                depth_pointcloud_folder = os.path.join(model_path, self.depth_pointcloud_folder_name)

        depth_pointcloud_files = sorted(glob.glob(os.path.join(depth_pointcloud_folder, '*.npz')))

        assert view_id is not None
        idx_img = view_id

        depth_pointcloud_file = depth_pointcloud_files[idx_img]

        # load npz
        depth_pointcloud_dict = np.load(depth_pointcloud_file)

        depth_pointcloud = depth_pointcloud_dict['pointcloud'].astype(np.float32)
        data = {
            None: depth_pointcloud
        }
        if self.transform is not None:
            data = self.transform(data)

        return data

class MixedInputField(Field):
    '''
        Mixed input field:
        
    Args:
        input_fields (list of str): input field names
        cfg (dict): config
    '''
    def __init__(self, input_field_names, mode, cfg, n_views=24):
        self.input_field_names = input_field_names
        self.mode = mode

        print('Mixed input field has', len(input_field_names), 'fields')
        print(input_field_names)

        self.input_fields = []
        self.n_views = n_views
        if mode == 'train':
            self.random_view = True
        else:
            self.random_view = False
        
        # get img's transform
        if mode == 'train' and cfg['data']['img_augment']:
            resize_op = transforms.RandomResizedCrop(
                cfg['data']['img_size'], (0.75, 1.), (1., 1.))
        else:
            resize_op = transforms.Resize((cfg['data']['img_size']))

        img_transform = transforms.Compose([
            resize_op, transforms.ToTensor(),
        ])


        # get all fields
        for fi in self.input_field_names:
            data_params = {}
            if fi == 'img':
                if 'img_extension' in cfg['data']:
                    data_params['extension'] = cfg['data']['img_extension']
                field = S_ImagesField(cfg['data']['img_folder'], transform=img_transform, **data_params)
            elif fi == 'camera':
                field = S_CameraField(cfg['data']['img_folder'])
            elif fi == 'depth':
                if 'absolute_depth' in cfg['data']:
                    data_params['absolute_depth'] = cfg['data']['absolute_depth']

                if 'with_minmax' in cfg['data']:
                    data_params['with_minmax'] = cfg['data']['with_minmax']

                if 'img_extension' in cfg['data']:
                    data_params['extension'] = cfg['data']['img_extension']
                field = S_DepthField('depth', transform=img_transform, **data_params)
            elif fi == 'mask':
                if 'img_extension' in cfg['data']:
                    data_params['extension'] = cfg['data']['img_extension']
                field = S_MaskField('mask', transform=img_transform, **data_params)
            elif fi == 'mask_flow':
                if 'img_extension' in cfg['data']:
                    data_params['extension'] = cfg['data']['img_extension']
                field = S_MaskFlowField('mask_flow', transform=img_transform, **data_params)
            elif fi == 'depth_pred':
                if 'absolute_depth' in cfg['data']:
                    data_params['absolute_depth'] = cfg['data']['absolute_depth']

                if 'with_minmax' in cfg['data']:
                    data_params['with_minmax'] = cfg['data']['with_minmax']

                if 'img_extension' in cfg['data']:
                    data_params['extension'] = cfg['data']['img_extension']
                field = S_DepthPredField(
                    cfg['data']['depth_pred_root'],
                    'depth_pred', transform=img_transform, 
                    **data_params
                )
            elif fi == 'depth_pointcloud':
                t_lst = []
                if 'depth_pointcloud_shuffle' in cfg['data'] and cfg['data']['depth_pointcloud_shuffle']:
                    t_lst.append(ShufflePointcloud())
                if 'depth_pointcloud_n' in cfg['data'] and cfg['data']['depth_pointcloud_n'] is not None:
                    t_lst.append(SubsamplePointcloud(cfg['data']['depth_pointcloud_n']))
                if 'depth_pointcloud_noise' in cfg['data'] and cfg['data']['depth_pointcloud_noise'] is not None:
                    t_lst.append(PointcloudNoise(cfg['data']['depth_pointcloud_noise']))
                if 'depth_pointcloud_dropout' in cfg['data'] and cfg['data']['depth_pointcloud_dropout'] != 0:
                    t_lst.append(PointcloudDropout(cfg['data']['depth_pointcloud_dropout']))
                pc_transform = transforms.Compose(t_lst)

                if mode == 'train' and 'depth_pointcloud_mix' in cfg['training']:
                    mixed = cfg['training']['depth_pointcloud_mix']
                else:
                    mixed = False

                field = S_DepthPointCloudField(
                    cfg['data']['depth_pointcloud_root'],
                    cfg['data']['depth_pointcloud_folder'],
                    transform = pc_transform,
                    mixed = mixed
                )
            elif fi == 'view_id':
                field = ViewIdField()
            elif fi == 'index':
                field = IndexField()
            elif fi == 'category':
                field = CategoryField()
            else:
                raise NotImplementedError

            self.input_fields.append(field)

    def gather(self, data, fi_name, fi_data):
        assert isinstance(data, dict)
        # inplace function

        if fi_name == 'img':
            data[None] = fi_data[None]
        elif fi_name == 'camera':
            data['world_mat'] = fi_data['world_mat']
            data['camera_mat'] = fi_data['camera_mat']
        elif fi_name == 'depth':
            data['depth'] = fi_data[None]
            if 'depth_min' in fi_data:
                data['depth_min'] = fi_data['depth_min']
                data['depth_max'] = fi_data['depth_max']
        elif fi_name == 'mask':
            data['mask'] = fi_data[None]
        elif fi_name == 'mask_flow':
            data['mask_flow'] = fi_data[None]
        elif fi_name == 'depth_pred':
            data['depth_pred'] = fi_data[None]
            if 'depth_pred_min' in fi_data:
                data['depth_pred_min'] = fi_data['depth_pred_min']
                data['depth_pred_max'] = fi_data['depth_pred_max']
        elif fi_name == 'depth_pointcloud':
            data['depth_pointcloud'] = fi_data[None]
        elif fi_name == 'view_id':
            data['view_id'] = fi_data
        elif fi_name == 'index':
            data['index'] = fi_data
        elif fi_name == 'category':
            data['category'] = fi_data
        else:
            raise NotImplementedError
        
        return data

    def load(self, model_path, idx, category, view_id=None):
        data = {}

        if view_id is not None:
            idx_img = view_id
        elif self.random_view:
            idx_img = random.randint(0, self.n_views - 1)
        else:
            idx_img = 0
        
        for i, fi in enumerate(self.input_fields):
            fi_data = fi.load(model_path, idx, category, view_id=idx_img)
            data = self.gather(data, self.input_field_names[i], fi_data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        # TODO: check
        return True