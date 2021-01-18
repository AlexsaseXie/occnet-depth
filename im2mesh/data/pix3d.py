import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import json
from im2mesh.data.core import Field
from im2mesh.data.transforms import SubsampleDepthPointcloud, PointcloudNoise
from im2mesh.data.fields import get_depth_image
from im2mesh.data.fields import IndexField

from scripts.pix3d_preprocess import utils as pix3d_utils

class Pix3dDataset(data.Dataset):
    def __init__(self, dataset_folder, fields,
        list_folder_name='generation_sampled_list', categories=None,
        no_except=True, transform=None,
        pix3d_root=None):
        '''
            categories: dir name
        '''
        self.dataset_folder = dataset_folder
        self.list_folder_name = list_folder_name

        self.fields = fields

        self.no_except = no_except
        self.transform = transform

        self.pix3d_root = pix3d_root

        # get list json
        if categories is not None:
            self.categories = categories
        else:
            self.categories = os.listdir(self.dataset_folder)

        self.all_images_infos = []
        for test_cls in self.categories:
            dataset_cls_root = os.path.join(self.dataset_folder, test_cls)
            dataset_list_path = os.path.join(dataset_cls_root, 'all_info.json')
            with open(dataset_list_path, 'r') as f:
                cls_dataset_infos = json.load(f)

            self.all_images_infos = self.all_images_infos + cls_dataset_infos

    def __getitem__(self, idx):
        image_info = self.get_info(idx)

        data = {}
        for field_name, field in self.fields.items():
            try:
                field_data = field.load(image_info, idx, self.pix3d_root)
            except Exception:
                if self.no_except:
                    print(
                        'Error occured when loading field %s of image %s'
                        % (field_name, image_info['img'])
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_info(self, idx):
        return self.all_images_infos[idx]

    def __len__(self):
        return len(self.all_images_infos)


# Fields
class Pix3d_S_ImagesField(Field):
    ''' Pix3d_S_Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
    '''
    def __init__(self, transform=None, resolution=224):
        self.transform = transform
        self.resolution = resolution

    def load(self, image_info, idx, pix3d_root=None):
        ''' Loads the data point.

        Args:
            image_info(dict)
            idx(int)
            pix3d_root(str)
        '''
        image_path = image_info['img_%d' % self.resolution]

        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        data = {
            None: image
        }

        return data

class Pix3d_S_CameraField(Field):
    ''' Pix3d_S_Camera Field.

    It is the field used for loading camera args.

    Args:
        resolution(int)
    '''
    def __init__(self, resolution=224):
        self.resolution = resolution

    def load(self, image_info, idx, pix3d_root=None):
        ''' Loads the data point.

        Args:
            image_info(dict)
            idx(int)
            pix3d_root(str)
        '''
        category = image_info['category']

        K = np.array(image_info['K_%d' % self.resolution]).astype(np.float32)
        Rt = np.array(image_info['Rt_%d' % self.resolution]).astype(np.float32)

        # need fix
        if category in ('chair', 'sofa', 'table'):
            fix_mat = np.array([
                [0,0,1,0],
                [0,1,0,0],
                [-1,0,0,0],
                [0,0,0,1]
            ])
        else:
            raise NotImplementedError
        
        Rt = np.dot(Rt, fix_mat)

        data = {}
        data['world_mat'] = Rt
        data['camera_mat'] = K

        return data

class Pix3d_S_MaskField(Field):
    ''' Pix3d_S_Mask Field.

    It is the field used for loading mask.

    Args:
        transform (list): list of transformations applied to loaded images
        resolution (int)
    '''
    def __init__(self, transform=None, resolution=224):
        self.transform = transform
        self.resolution = resolution

    def load(self, image_info, idx, pix3d_root=None):
        ''' Loads the data point.

        Args:
            image_info(dict)
            idx(int)
            pix3d_root(str)
        '''
        mask_path = image_info['mask_%d' % self.resolution]

        depth_mask = Image.open(mask_path).convert('1')
        if self.transform is not None:
            depth_mask = self.transform(depth_mask)

        data = {
            None: depth_mask
        }

        return data

class Pix3d_S_DepthPredField(Field):
    ''' Pix3d_S_DepthPred Field.

    It is the field used for loading predicted depth maps.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
    '''
    def __init__(self, depth_pred_root=None, depth_pred_folder_name='depth_pred',
                  transform=None, absolute_depth=True, with_minmax=False):
        self.depth_pred_root = depth_pred_root
        self.depth_pred_folder_name = depth_pred_folder_name
        self.transform = transform
        self.absolute_depth = absolute_depth
        self.with_minmax = with_minmax

    def load(self, image_info, idx, pix3d_root=None):
        ''' Loads the data point.

        Args:
            image_info(dict)
            idx(int)
            pix3d_root(str)
        '''
        assert self.depth_pred_root is not None

        image_category = image_info['category']
        image_name = pix3d_utils.get_image_name(image_info)

        depth_pred_folder = os.path.join(self.depth_pred_root, image_category, image_name, self.depth_pred_folder_name)

        depth_pred_image, depth_pred_min, depth_pred_max = get_depth_image(depth_pred_folder, 0, transform=self.transform, extension='png')
            
        if self.absolute_depth:
            depth_pred_image = depth_pred_image * (depth_pred_max - depth_pred_min) + depth_pred_min

        data = {
            None: depth_pred_image
        }

        if self.with_minmax:
            data['depth_pred_min'] = depth_pred_min
            data['depth_pred_max'] = depth_pred_max

        return data

class Pix3d_S_DepthPointCloudField(Field):
    ''' Depth Point Cloud Field.

    It is the field used for loading depth point cloud.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
    '''
    def __init__(self, depth_pointcloud_root=None, depth_pointcloud_folder_name='depth_pointcloud',
                  transform=None):
        self.depth_pointcloud_root = depth_pointcloud_root
        self.depth_pointcloud_folder_name = depth_pointcloud_folder_name
        self.transform = transform

    def load(self, image_info, idx, pix3d_root=None):
        ''' Loads the data point.

        Args:
            image_info(dict)
            idx(int)
            pix3d_root(str)
        '''
        image_category = image_info['category']
        image_name = pix3d_utils.get_image_name(image_info)

        assert self.depth_pointcloud_root is not None
        depth_pointcloud_folder = os.path.join(self.depth_pointcloud_root, image_category, image_name, self.depth_pointcloud_folder_name)

        depth_pointcloud_file = os.path.join(depth_pointcloud_folder, '00.npz')

        # load npz
        depth_pointcloud_dict = np.load(depth_pointcloud_file)

        depth_pointcloud = depth_pointcloud_dict['pointcloud'].astype(np.float32)
        if self.transform is not None:
            depth_pointcloud = self.transform(depth_pointcloud)

        data = {
            None: depth_pointcloud
        }

        return data

class Pix3d_MixedInputField(Field):
    '''
        Mixed input field:
        
    Args:
        input_fields (list of str): input field names
        cfg (dict): config
    '''
    def __init__(self, input_field_names, mode, cfg):
        self.input_field_names = input_field_names
        self.mode = mode

        print('Mixed input field has', len(input_field_names), 'fields')
        print(input_field_names)

        self.input_fields = []
        
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
                field = Pix3d_S_ImagesField(transform=img_transform, resolution=cfg['data']['img_size'])
            elif fi == 'camera':
                field = Pix3d_S_CameraField(resolution=cfg['data']['img_size'])
            #elif fi == 'depth':
            #    if 'absolute_depth' in cfg['data']:
            #        data_params['absolute_depth'] = cfg['data']['absolute_depth']

            #    if 'with_minmax' in cfg['data']:
            #        data_params['with_minmax'] = cfg['data']['with_minmax']

            #    if 'img_extension' in cfg['data']:
            #        data_params['extension'] = cfg['data']['img_extension']
            #    field = Pix3d_S_DepthField('depth', transform=img_transform, **data_params)
            elif fi == 'mask':
                field = Pix3d_S_MaskField(transform=img_transform, resolution=cfg['data']['img_size'])
            #elif fi == 'mask_flow':
            #    if 'img_extension' in cfg['data']:
            #        data_params['extension'] = cfg['data']['img_extension']
            #    field = Pix3d_S_MaskFlowField('mask_flow', transform=img_transform, **data_params)
            elif fi == 'depth_pred':
                if 'absolute_depth' in cfg['data']:
                    data_params['absolute_depth'] = cfg['data']['absolute_depth']

                if 'with_minmax' in cfg['data']:
                    data_params['with_minmax'] = cfg['data']['with_minmax']

                field = Pix3d_S_DepthPredField(
                    cfg['data']['depth_pred_root'],
                    'depth_pred', transform=img_transform, 
                    **data_params
                )
            elif fi == 'depth_pointcloud':
                t_lst = []
                if 'depth_pointcloud_n' in cfg['data'] and cfg['data']['depth_pointcloud_n'] is not None:
                    t_lst.append(SubsampleDepthPointcloud(cfg['data']['depth_pointcloud_n']))
                if 'depth_pointcloud_noise' in cfg['data'] and cfg['data']['depth_pointcloud_noise'] is not None:
                    t_lst.append(PointcloudNoise(cfg['data']['depth_pointcloud_noise']))
                pc_transform = transforms.Compose(t_lst)

                #if 'depth_pointcloud_mix' in cfg['training']:
                #    mixed = cfg['training']['depth_pointcloud_mix']
                #else:
                #    mixed = False

                field = Pix3d_S_DepthPointCloudField(
                    cfg['data']['depth_pointcloud_root'],
                    cfg['data']['depth_pointcloud_folder'],
                    transform = pc_transform,
                )
            #elif fi == 'view_id':
            #    field = ViewIdField()
            elif fi == 'index':
                field = IndexField()
            #elif fi == 'category':
            #    field = CategoryField()
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

    def load(self, image_info, idx, pix3d_root=None):
        data = {}
        
        for i, fi in enumerate(self.input_fields):
            fi_data = fi.load(image_info, idx, pix3d_root)
            data = self.gather(data, self.input_field_names[i], fi_data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        # TODO: check
        return True   