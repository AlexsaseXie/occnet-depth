import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from im2mesh.data.core import Field
from im2mesh.utils import binvox_rw
import torch
from torchvision import transforms
import h5py
from tqdm import tqdm

# read img functions
def get_depth_image(depth_folder, idx_img, transform=None, extension='png'):
    depth_files = sorted(glob.glob(os.path.join(depth_folder, '*.%s' % extension)))

    depth_range_file = os.path.join(depth_folder, 'depth_range.txt')
    with open(depth_range_file,'r') as f:
        txt = f.readlines()
        depth_range = txt[idx_img].split(' ')
    depth_min = float(depth_range[0])
    depth_max = float(depth_range[1])
    depth_unit = float(depth_range[2])

    depth_filename = depth_files[idx_img]

    depth_image = Image.open(depth_filename).convert('L')
    if transform is not None:
        depth_image = transform(depth_image)

    #depth_image = depth_image * (depth_max - depth_min) + depth_min
    #depth_image = depth_image / depth_unit
    return depth_image, depth_min / depth_unit, depth_max / depth_unit

def get_mask(mask_folder, idx_img, transform=None, extension='png'):
    mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.%s' % extension)))
    mask_filename = mask_files[idx_img]

    depth_mask = Image.open(mask_filename).convert('1')
    if transform is not None:
        depth_mask = transform(depth_mask)

    return depth_mask

def get_image(img_folder, idx_img, transform=None, extension='png'):
    img_files = sorted(glob.glob(os.path.join(img_folder, '*.%s' % extension)))
    img_filename = img_files[idx_img]

    image = Image.open(img_filename).convert('RGB')
    if transform is not None:
        image = transform(image)
    
    return image

def get_mask_flow(mask_flow_folder, idx_img, transform=None, extension='png'):
    mask_flow_files = sorted(glob.glob(os.path.join(mask_flow_folder, '*.%s' % extension)))
    mask_flow_filename = mask_flow_files[idx_img]

    mask_flow = Image.open(mask_flow_filename).convert('L')
    if transform is not None:
        mask_flow = transform(mask_flow)

    return mask_flow


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class ViewIdField(Field):
    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the viewid field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if view_id is None:
            raise Exception('When loading ViewIdField, viewid is None!')
        return view_id

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class CategoryField(Field):
    ''' Basic category field.'''
    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the category field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return category

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

class ImagesField(Field):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        if view_id is not None:
            idx_img = view_id
        elif self.random_view:
            idx_img = random.randint(0, len(files)-1)
        else:
            idx_img = 0
        
        image = get_image(folder, idx_img, transform=self.transform, extension=self.extension)

        data = {
            None: image
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete

class MultiImageField(Field):
    ''' MultiImage Field.

    It is the field used for loading images.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, folder_name, n_views=3, transform=None,
                 extension='jpg', random_view=True, with_camera=False):
        self.folder_name = folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera
        self.n_views = n_views

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        
        if self.random_view:
            choices = range(len(files) - 1)
            idx_img = random.sample(choices, self.n_views)
        else:
            idx_img = list(range(self.n_views))

        img_set = []
        for idx in idx_img:
            filename = files[int(idx)]

            image = Image.open(filename).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)

            img_set.append(image)

        data = {
            None: torch.stack(img_set)
        }

        if self.with_camera:
            camera_file = os.path.join(folder, 'cameras.npz')
            camera_dict = np.load(camera_file)

            Rts = []
            Ks = []
            for idx in idx_img:
                Rt = camera_dict['world_mat_%d' % int(idx)].astype(np.float32)
                Rts.append(Rt)
                K = camera_dict['camera_mat_%d' % int(idx)].astype(np.float32)
                Ks.append(Rt)

            data['world_mat'] = np.array(Rts)
            data['camera_mat'] = np.array(Ks)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete


# 3D Fields
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, transform=None, with_transforms=False, unpackbits=False, input_range=None):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms
        self.unpackbits = unpackbits
        self.input_range = input_range
        print('Points_field:', self.file_name)

    def preload(self, dataset_folder, models_info):
        '''
            preload function
        '''
        print('Preloading PointsField')
        self.model_data = []
        for model_info in tqdm(models_info):
            model = model_info['model']
            category = model_info['category']
            model_path = os.path.join(dataset_folder, category, model)

            file_path = os.path.join(model_path, self.file_name)
            points_dict = np.load(file_path)

            points = points_dict['points']
            if points.dtype == np.float16:
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)

            occupancies = points_dict['occupancies']
            if self.unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)

            data = {
                None: points,
                'occ': occupancies,
            }

            if self.with_transforms:
                data['loc'] = points_dict['loc'].astype(np.float32)
                data['scale'] = points_dict['scale'].astype(np.float32)

            self.model_data.append(data)
        
        self.preloaded = True

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if getattr(self, 'preloaded', False):
            return load_preloaed(idx)

        file_path = os.path.join(model_path, self.file_name)

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)
        else:
            points = points.astype(np.float32)

        occupancies = points_dict['occupancies']

        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        if self.input_range is not None:
            points = points[self.input_range[0]: self.input_range[1]]
            occupancies = occupancies[self.input_range[0]: self.input_range[1]]

        #surface point occ = 0.5
        #occupancies[(occupancies > 0) & (occupancies < 1)] = 0.5

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.with_transforms:
            data['loc'] = points_dict['loc'].astype(np.float32)
            data['scale'] = points_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def load_preloaed(self, idx):
        data = self.model_data[idx].copy()

        if self.transform is not None:
            data = self.transform(data)

        return data

class PointsH5Field(Field):
    ''' Point using h5 Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, subsample_n=None, with_transforms=False, input_range=None, chunked=True, random_choice=False):
        self.file_name = file_name

        if subsample_n is None:
            chunked = False
        self.chunked = chunked

        if subsample_n is not None:
            self.N = subsample_n
            if input_range is not None:
                assert self.N < input_range[1] - input_range[0]
        else:
            if input_range is not None:
                self.N = input_range[1] - input_range[0]
            else:
                self.N = 0

        self.random_choice = random_choice
        self.input_range = input_range
        self.with_transforms = with_transforms

        print('Points h5 field:', self.file_name)

    def preload(self, dataset_folder, models_info):
        '''
            preload function
        '''
        print('Preloading PointsH5Field')
        self.model_data = []
        for model_info in tqdm(models_info):
            model = model_info['model']
            category = model_info['category']
            model_path = os.path.join(dataset_folder, category, model)

            file_path = os.path.join(model_path, self.file_name)
            with h5py.File(file_path, 'r') as h5f:
                self.total_length = h5f['points'].shape[0]
                if self.N == 0:
                    self.N = self.total_length

                if self.input_range is None:
                    self.input_range = [0, self.total_length]

                points = h5f['points'][()]
                occupancies = h5f['occupancies'][()].astype(np.float32)
                if points.dtype == np.float16:
                    points = points.astype(np.float32)
                    points += 1e-4 * np.random.randn(*points.shape)
                else:
                    points = points.astype(np.float32)

                data = {
                    None: points,
                    'occ': occupancies,
                }

                if self.with_transforms:
                    data['loc'] = h5f['loc'].astype(np.float32)
                    data['scale'] = h5f['scale'].astype(np.float32)

                self.model_data.append(data)
        
        self.preloaded = True

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if getattr(self, 'preloaded', False):
            return self.load_preloaded(idx)

        if self.chunked:
            return self.load_chunked(model_path, idx, category, view_id)

        file_path = os.path.join(model_path, self.file_name)

        with h5py.File(file_path, 'r') as h5f:
            if self.N != 0:
                N = self.N
            else:
                N = h5f['points'].shape[0]

            if self.input_range is not None:
                low = self.input_range[0]
                high = self.input_range[1]
            else:
                low = 0
                high = h5f['points'].shape[0]
        
            if N < high - low:
                if self.random_choice:
                    #random choice
                    idx = np.random.choice(range(low, high), N, False)
                    idx.sort()
                    pt_idx = np.s_[idx, :]
                    occ_idx = list(idx)
                else:
                    #continuous
                    start = np.random.randint(low, high - N)
                    pt_idx = np.s_[start: start + N, :]
                    occ_idx = np.s_[start: start + N]
            else:
                pt_idx = np.s_[low:high,:]
                occ_idx = np.s_[low:high]

            points = h5f['points'][pt_idx]
            occupancies = h5f['occupancies'][occ_idx].astype(np.float32)
            if points.dtype == np.float16:
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)
            
                
            data = {
                None: points,
                'occ': occupancies,
            }

            if self.with_transforms:
                data['loc'] = h5f['loc'][:].astype(np.float32)
                data['scale'] = h5f['scale'][()].astype(np.float32)

        return data

    def load_chunked(self, model_path, idx, category, view_id=None):
        assert self.N != 0
        # input_range is not used here

        file_path = os.path.join(model_path, self.file_name)

        with h5py.File(file_path, 'r') as h5f:
            total_length = h5f['points'].shape[0]
            total_chunk_count = total_length // self.N
            if total_length % self.N != 0:
                total_chunk_count += 1

            choice = np.random.randint(0, total_chunk_count)
            if choice == total_chunk_count - 1:
                pt_idx = np.s_[total_length - self.N: total_length, :]
                occ_idx = np.s_[total_length - self.N: total_length]
            else:            
                pt_idx = np.s_[choice * self.N: (choice+1) * self.N, :]
                occ_idx = np.s_[choice * self.N: (choice+1) * self.N]

            points = h5f['points'][pt_idx]
            occupancies = h5f['occupancies'][occ_idx].astype(np.float32)
            if points.dtype == np.float16:
                # break symmetric
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)

            data = {
                None: points,
                'occ': occupancies,
            }

            if self.with_transforms:
                data['loc'] = h5f['loc'][:].astype(np.float32)
                data['scale'] = h5f['scale'][()].astype(np.float32)

        return data

    def load_preloaded(self, idx):
        data = self.model_data[idx].copy()

        low = self.input_range[0]
        high = self.input_range[1]

        if self.chunked:
            assert self.N < self.total_length
            total_chunk_count = self.total_length // self.N
            if total_length % self.N != 0:
                total_chunk_count += 1

            choice = np.random.randint(0, total_chunk_count)
            if choice == total_chunk_count - 1:
                pt_idx = np.s_[total_length - self.N: total_length, :]
                occ_idx = np.s_[total_length - self.N: total_length]
            else:            
                pt_idx = np.s_[choice * self.N: (choice+1) * self.N, :]
                occ_idx = np.s_[choice * self.N: (choice+1) * self.N]
        else:
            if self.N < high - low:
                if self.random_choice:
                    #random choice
                    idx = np.random.choice(range(low, high), self.N, False)
                    idx.sort()
                    pt_idx = np.s_[idx, :]
                    occ_idx = list(idx)
                else:
                    #continuous
                    start = np.random.randint(low, high - self.N)
                    pt_idx = np.s_[start: start + self.N, :]
                    occ_idx = np.s_[start: start + self.N]
            else:
                pt_idx = np.s_[low:high,:]
                occ_idx = np.s_[low:high]

        data.update({
            None: data[None][pt_idx],
            'occ': data['occ'][occ_idx]
        })

        return data


class SdfH5Field(Field):
    ''' Point SDF using h5 Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the
            points tensor
        with_transforms (bool): whether scaling and rotation data should be
            provided

    '''
    def __init__(self, file_name, subsample_n=None, with_transforms=False, input_range=None, chunked=True, random_choice=False):
        self.file_name = file_name

        if subsample_n is None:
            chunked = False
        self.chunked = chunked

        if subsample_n is not None:
            self.N = subsample_n
            if input_range is not None:
                assert self.N < input_range[1] - input_range[0]
        else:
            if input_range is not None:
                self.N = input_range[1] - input_range[0]
            else:
                self.N = 0

        self.random_choice = random_choice
        self.input_range = input_range
        self.with_transforms = with_transforms

        print('Sdf h5 field:', self.file_name)

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.chunked:
            return self.load_chunked(model_path, idx, category, view_id)

        file_path = os.path.join(model_path, self.file_name)

        with h5py.File(file_path, 'r') as h5f:
            if self.N != 0:
                N = self.N
            else:
                N = h5f['points'].shape[0]

            if self.input_range is not None:
                low = self.input_range[0]
                high = self.input_range[1]
            else:
                low = 0
                high = h5f['points'].shape[0]
        
            if N < high - low:
                if self.random_choice:
                    #random choice
                    idx = np.random.choice(range(low, high), N, False)
                    idx.sort()
                    pt_idx = np.s_[idx, :]
                    sdf_idx = list(idx)
                else:
                    #continuous
                    start = np.random.randint(low, high - N)
                    pt_idx = np.s_[start: start + N, :]
                    sdf_idx = np.s_[start: start + N]
            else:
                pt_idx = np.s_[low:high,:]
                sdf_idx = np.s_[low:high]

            points = h5f['points'][pt_idx]
            sdf = h5f['sdf'][sdf_idx].astype(np.float32)
            surface_flag = h5f['surface_points_flag'][sdf_idx].astype(np.float32)

            if points.dtype == np.float16:
                # break symmetric
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)

            data = {
                None: points,
                'sdf': sdf,
                'surface_flag': surface_flag
            }

            if self.with_transforms:
                data['loc'] = h5f['loc'][:].astype(np.float32)
                data['scale'] = h5f['scale'][()].astype(np.float32)

        return data

    def load_chunked(self, model_path, idx, category, view_id=None):
        assert self.N != 0
        #input_range is not used

        file_path = os.path.join(model_path, self.file_name)

        with h5py.File(file_path, 'r') as h5f:
            total_length = h5f['points'].shape[0]
            total_chunk_count = total_length // self.N
            if total_length % self.N != 0:
                total_chunk_count += 1

            choice = np.random.randint(0, total_chunk_count)
            if choice == total_chunk_count - 1:
                pt_idx = np.s_[total_length - self.N: total_length, :]
                sdf_idx = np.s_[total_length - self.N: total_length]
            else:            
                pt_idx = np.s_[choice * self.N: (choice+1) * self.N, :]
                sdf_idx = np.s_[choice * self.N: (choice+1) * self.N]

            points = h5f['points'][pt_idx]
            sdf = h5f['sdf'][sdf_idx].astype(np.float32)
            surface_flag = h5f['surface_points_flag'][sdf_idx].astype(np.float32)

            if points.dtype == np.float16:
                # break symmetric
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)

            data = {
                None: points,
                'sdf': sdf,
                'surface_flag': surface_flag
            }

            if self.with_transforms:
                data['loc'] = h5f['loc'][:].astype(np.float32)
                data['scale'] = h5f['scale'][()].astype(np.float32)

        return data

class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        with_transforms (bool): whether scaling and rotation dat should be
            provided
    '''
    def __init__(self, file_name, transform=None, with_transforms=False):
        self.file_name = file_name
        self.transform = transform
        self.with_transforms = with_transforms

    def preload(self, dataset_folder, models_info):
        '''
            preload function
        '''
        print('Preloading PointsField')
        self.model_data = []
        for model_info in tqdm(models_info):
            model = model_info['model']
            category = model_info['category']
            model_path = os.path.join(dataset_folder, category, model)

            file_path = os.path.join(model_path, self.file_name)
            pointcloud_dict = np.load(file_path)

            points = pointcloud_dict['points'].astype(np.float32)
            normals = pointcloud_dict['normals'].astype(np.float32)

            data = {
                None: points,
                'normals': normals,
            }

            if self.with_transforms:
                data['loc'] = pointcloud_dict['loc'].astype(np.float32)
                data['scale'] = pointcloud_dict['scale'].astype(np.float32)

            self.model_data.append(data)
        
        self.preloaded = True

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if getattr(self, 'preloaded', False):
            return load_preloaded(idx)

        file_path = os.path.join(model_path, self.file_name)

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        data = {
            None: points,
            'normals': normals,
        }

        if self.with_transforms:
            data['loc'] = pointcloud_dict['loc'].astype(np.float32)
            data['scale'] = pointcloud_dict['scale'].astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

    def load_preloaed(self, idx):
        data = self.model_data[idx].copy()

        if self.transform is not None:
            data = self.transform(data)

        return data

# NOTE: this will produce variable length output.
# You need to specify collate_fn to make it work with a data laoder
class MeshField(Field):
    ''' Mesh field.

    It provides the field used for mesh data. Note that, depending on the
    dataset, it produces variable length output, so that you need to specify
    collate_fn to make it work with a data loader.

    Args:
        file_name (str): file name
        transform (list): list of transforms applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        mesh = trimesh.load(file_path, process=False)
        if self.transform is not None:
            mesh = self.transform(mesh)

        data = {
            'verts': mesh.vertices,
            'faces': mesh.faces,
        }

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

# Depth related fields
class ImagesWithDepthField(Field):
    ''' Image With Depth Field.

    It is the field used for loading images and depth.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, img_folder_name='img', depth_folder_name='depth', mask_folder_name='mask', transform=None,
                 extension='png', random_view=True, with_camera=False, absolute_depth=True, with_minmax=False):
        self.img_folder_name = img_folder_name
        self.depth_folder_name = depth_folder_name
        self.mask_folder_name = mask_folder_name
        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera
        self.absolute_depth = absolute_depth
        self.with_minmax = with_minmax

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        img_folder = os.path.join(model_path, self.img_folder_name)
        img_files = sorted(glob.glob(os.path.join(img_folder, '*.%s' % self.extension)))
        depth_folder = os.path.join(model_path, self.depth_folder_name)
        mask_folder = os.path.join(model_path, self.mask_folder_name)

        if view_id is not None:
            idx_img = view_id
        elif self.random_view:
            idx_img = random.randint(0, len(img_files)-1)
        else:
            idx_img = 0

        image = get_image(img_folder, idx_img, transform=self.transform, extension=self.extension)
        depth_image, depth_min, depth_max = get_depth_image(depth_folder, idx_img, transform=self.transform, extension=self.extension)
        if self.absolute_depth:
            depth_image = depth_image * (depth_max - depth_min) + depth_min

        depth_mask = get_mask(mask_folder, idx_img, transform=self.transform, extension=self.extension)

        data = {
            None: image,
            'depth': depth_image,
            'mask': depth_mask
        }

        if self.with_minmax:
            data['depth_min'] = depth_min
            data['depth_max'] = depth_max

        if self.with_camera:
            camera_file = os.path.join(img_folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.img_folder_name in files) & (self.depth_folder_name in files)
        # TODO: check camera
        return complete

class DepthPredictedField(Field):
    ''' Depth Field.

    It is the field used for loading depth maps.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
        with_camera (bool): whether camera data should be provided
    '''
    def __init__(self, img_folder_name='img', depth_folder_name='depth', mask_folder_name='mask', 
                  depth_pred_root=None, depth_pred_folder_name='depth_pred',
                  transform=None,extension='png', random_view=True, with_camera=False,
                  absolute_depth=True, with_minmax=False, with_img=False):
        self.img_folder_name = img_folder_name
        self.depth_folder_name = depth_folder_name
        self.mask_folder_name = mask_folder_name

        self.depth_pred_root = depth_pred_root
        self.depth_pred_folder_name = depth_pred_folder_name

        self.transform = transform
        self.extension = extension
        self.random_view = random_view
        self.with_camera = with_camera
        self.absolute_depth = absolute_depth
        self.with_minmax = with_minmax
        self.with_img = with_img

    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        img_folder = os.path.join(model_path, self.img_folder_name)
        depth_folder = os.path.join(model_path, self.depth_folder_name)
        mask_folder = os.path.join(model_path, self.mask_folder_name)
        img_files = sorted(glob.glob(os.path.join(img_folder, '*.%s' % self.extension)))

        if view_id is not None:
            idx_img = view_id
        elif self.random_view:
            idx_img = random.randint(0, len(img_files)-1)
        else:
            idx_img = 0

        depth_image, depth_min, depth_max = get_depth_image(depth_folder, idx_img, transform=self.transform, extension=self.extension)
        depth_mask = get_mask(mask_folder, idx_img, transform=self.transform, extension=self.extension)

        # self.depth_pred_root is not None:
        paths = model_path.split('/')
        depth_pred_folder = os.path.join(self.depth_pred_root, 
            paths[-2], paths[-1], 
            self.depth_pred_folder_name)
        depth_pred_image, depth_pred_min, depth_pred_max = get_depth_image(depth_pred_folder, idx_img, transform=self.transform, extension=self.extension)
            
        if self.absolute_depth:
            depth_image = depth_image * (depth_max - depth_min) + depth_min
            depth_pred_image = depth_pred_image * (depth_pred_max - depth_pred_min) + depth_pred_min

        data = {
            'depth': depth_image,
            'mask': depth_mask,
            'depth_pred': depth_pred_image
        }

        if self.with_img:
            img = get_image(img_folder, idx_img, transform=self.transform, extension=self.extension)
            data[None] = img

        if self.with_minmax:
            data['depth_min'] = depth_min
            data['depth_max'] = depth_max
            data['depth_pred_min'] = depth_pred_min
            data['depth_pred_max'] = depth_pred_max

        if self.with_camera:
            camera_file = os.path.join(img_folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.img_folder_name in files) & (self.depth_folder_name in files)
        # TODO: check camera
        return complete

class DepthPointCloudField(Field):
    ''' Depth Point Cloud Field.

    It is the field used for loading depth point cloud.

    Args:
        folder_name (str): folder name
        transform (list): list of transformations applied to loaded images
        extension (str): image extension
        random_view (bool): whether a random view should be used
    '''
    def __init__(self, depth_pointcloud_root=None, depth_pointcloud_folder_name='depth_pointcloud',
                  transform=None, random_view=True, with_camera=False, img_folder_name='img', 
                  with_mask=False, mask_folder_name='mask', mask_flow_folder_name='mask_flow',
                  extension='png', img_transform=None,
                  with_depth_pred=False, depth_pred_root='same_as_depth_pointcloud', 
                  depth_pred_folder_name='depth_pred'):
        self.depth_pointcloud_root = depth_pointcloud_root
        self.depth_pointcloud_folder_name = depth_pointcloud_folder_name

        self.transform = transform
        self.random_view = random_view
        self.with_camera = with_camera
        self.img_folder_name = img_folder_name
        
        self.with_mask = with_mask
        if self.with_mask:
            self.mask_folder_name = mask_folder_name
            self.mask_flow_folder_name = mask_flow_folder_name
            self.extension = extension
            self.img_transform = img_transform

        self.with_depth_pred = with_depth_pred
        if self.with_depth_pred:
            if depth_pred_root == 'same_as_depth_pointcloud':
                self.depth_pred_root = self.depth_pointcloud_root
            else:
                self.depth_pred_root = depth_pred_root
            self.depth_pred_folder_name = depth_pred_folder_name


    def load(self, model_path, idx, category, view_id=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.depth_pointcloud_root is not None:
            paths = model_path.split('/')
            depth_pointcloud_folder = os.path.join(self.depth_pointcloud_root, paths[-2], paths[-1], self.depth_pointcloud_folder_name)
        else:
            depth_pointcloud_folder = os.path.join(model_path, self.depth_pointcloud_folder_name)

        depth_pointcloud_files = sorted(glob.glob(os.path.join(depth_pointcloud_folder, '*.npz')))

        if view_id is not None:
            idx_img = view_id
        elif self.random_view:
            idx_img = random.randint(0, len(depth_pointcloud_files)-1)
        else:
            idx_img = 0

        depth_pointcloud_file = depth_pointcloud_files[idx_img]

        # load npz
        depth_pointcloud_dict = np.load(depth_pointcloud_file)

        depth_pointcloud = depth_pointcloud_dict['pointcloud'].astype(np.float32)
        if self.transform is not None:
            depth_pointcloud = self.transform(depth_pointcloud)

        data = {
            'depth_pointcloud': depth_pointcloud
        }

        if self.with_camera:
            img_folder = os.path.join(model_path, self.img_folder_name)
            camera_file = os.path.join(img_folder, 'cameras.npz')
            camera_dict = np.load(camera_file)
            Rt = camera_dict['world_mat_%d' % idx_img].astype(np.float32)
            K = camera_dict['camera_mat_%d' % idx_img].astype(np.float32)
            data['world_mat'] = Rt
            data['camera_mat'] = K

        if self.with_mask:
            mask_folder = os.path.join(model_path, self.mask_folder_name)
            depth_mask = get_mask(mask_folder, idx_img, transform=self.img_transform, extension=self.extension)
            data['mask'] = depth_mask

            mask_flow_folder = os.path.join(model_path, self.mask_flow_folder_name)
            mask_flow = get_mask_flow(mask_flow_folder, idx_img, transform=self.img_transform, extension=self.extension)
            data['mask_flow'] = mask_flow

        if self.with_depth_pred:
            if self.depth_pred_root is not None:
                paths = model_path.split('/')
                depth_pred_folder = os.path.join(self.depth_pred_root, paths[-2], paths[-1], self.depth_pred_folder_name)
            else:
                depth_pred_folder = os.path.join(model_path, self.depth_pred_folder_name)
            depth_pred_image, depth_pred_min, depth_pred_max = get_depth_image(depth_pred_folder, idx_img, transform=self.img_transform, extension=self.extension)
            # must be absolute depth
            depth_pred_image = depth_pred_image * (depth_pred_max - depth_pred_min) + depth_pred_min
            data['depth_pred'] = depth_pred_image
            

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = True
        return complete

