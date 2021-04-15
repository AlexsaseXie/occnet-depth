import torch
from torch import nn
import numpy as np
from PIL import Image
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

# for single numpy image
# default: sensor's width = 32 mm
# focal length = 35 mm
# image width = image height = 224

# !!!NOTICE: 
# Acctually later I find that here's a bug that the x, y axes should be transposed.
# In my previous implementation I tranpose the points before sending into later point cloud completion network(MSN).
# As img_h == img_w in our cases, the following methods are still Okay. 
# But with img_w != img_h, bugs should occur.
class DepthToPCNp:
    def __init__(self, focal_div_sensor=35./32.):
        self.focal_div_sensor = focal_div_sensor
        
        self.grid_x, self.grid_y = np.mgrid[:224, :224]

    def sample_resize(self, depth, mask, n=2048):
        '''
            depth : img_w * img_h numpy.ndarray float32
            mask : img_w * img_h numpy.ndarray bool

            returns depth: img_w * img_h numpy.ndarray float32
                    mask: img_w * img_h numpy.ndarray bool
        '''
        img_w = depth.shape[0]
        img_h = depth.shape[1]
        current_count = mask.sum()

        factor = np.ceil(np.sqrt(n / current_count))
       
        if current_count < n:
            mask_img = mask.astype(np.uint8) * 128
            mask_img = Image.fromarray(mask_img)
            
        while current_count < n:
            #print(current_count, n, ',factor:', factor)
            new_size = (int(img_w * factor), int(img_h * factor))
            
            # transfer to uint8
            interior_mask_img = mask_img.resize(new_size, Image.BILINEAR)
            interior_mask_img = interior_mask_img.point(lambda i: i >= 128, '1')
            border_mask_img = mask_img.resize(new_size, Image.NEAREST)
            border_mask_img = border_mask_img.point(lambda i: i >= 128, '1')

            mask = np.array(interior_mask_img)
            border_mask = np.array(border_mask_img)
            border_mask = np.logical_not(mask) & border_mask
            #print('border count = ', border_mask.sum())            
            mask[border_mask] = True

            current_count = mask.sum()
            #print('current count = ', current_count)
            if current_count >= n:
                depth = (1.0 / depth).astype(np.float32)
                new_depth_img = Image.fromarray(depth, 'F')
                interior_depth_img = new_depth_img.resize(new_size, Image.BILINEAR)
                border_depth_img = new_depth_img.resize(new_size, Image.NEAREST)
                depth = 1.0 / np.array(interior_depth_img)

                border_depth = 1.0 / np.array(border_depth_img)
                depth[border_mask] = border_depth[border_mask]
                break
            else:
                #print('current_count:%d' % current_count, ',factor:', factor)
                factor = factor * 2.
           
            assert False

        return depth, mask

    def back_projection(self, depth, unit=1., align_corners=False):
        '''
            depth : img_w * img_h np.ndarray
            returns pc_xyz: img_w * img_h * 3
        '''
        img_h = depth.shape[0]
        img_w = depth.shape[1]
        u_0 = img_w / 2
        v_0 = img_h / 2
        f_u = img_w * self.focal_div_sensor
        f_v = f_u

        if img_w == 224 and img_h == 224:
            grid_x = self.grid_x
            grid_y = self.grid_y
        else:
            grid_x, grid_y = np.mgrid[:img_w, :img_h]

        # important update: 
        # by default, we should treat the center of each pixel to lie at [x+0.5, y+0.5]
        # the pixel covers [x, x+1] \times [y, y+1]

        # !!!BUG occurs: need transpose the depth first
        if (not align_corners):
            pc_x = (grid_x - u_0) * depth / f_u
            pc_y = (grid_y - v_0) * depth / f_v
        else:
            pc_x = (grid_x - u_0 + 0.5) * depth / f_u
            pc_y = (grid_y - v_0 + 0.5) * depth / f_v
        pc_xyz = np.stack((pc_x, pc_y, depth - unit), axis=2)

        return pc_xyz

    def mask_sample(self, pc_xyz, mask, n=None, sample_strategy='random'):
        '''
            mask : img_w * img_h
            pc_xyz : img_w * img_h * 3

            returns pts : n * 3 points
        '''
        assert pc_xyz.shape[0] == mask.shape[0]
        assert pc_xyz.shape[1] == mask.shape[1]

        pts = pc_xyz[mask]
        if n is not None:
            if pts.shape[0] < n:
                choice = np.random.choice(range(pts.shape[0]), size=n - pts.shape[0], replace=True)
                pts_append = pts[choice]
                # concat
                pts = np.concatenate((pts, pts_append), axis=0)
            else:
                # assert (pts.shape[0] >= n):
                if sample_strategy == 'random':
                    choice = np.random.choice(range(pts.shape[0]), size=n, replace=False)
                    pts = pts[choice]
                elif sample_strategy == 'fps':
                    pts_torch = torch.from_numpy(pts).unsqueeze(0).cuda() # cuda tensor: 1 * n_pts * 3
                    pts_torch_tranposed = pts_torch.transpose(1, 2) # 1 * 3 * n_pts

                    idx = furthest_point_sample(pts_torch, n)

                    pts_torch_transposed = gather_operation(pts_torch_tranposed, idx) # cuda tensor: 1 * 3 * n
                    pts_torch = pts_torch_tranposed.transpose(1, 2) # 1 * n * 3
                    pts = pts_torch.numpy().squeeze(0)
                else:
                    raise NotImplementedError
                
        
        return pts

    def work(self, depth_img, mask_img, depth_min, depth_max, 
            # resize related params
            resize=True,
            # back projection related params
            unit=1., align_corners=False,
            # mask sample related params
            n=2048, sample_strategy='random'
        ):
        '''
            depth_img : PIL Image (mode L)
            mask_img : PIL Image (mode 1)
            depth_min : float
            depth_max : float 
        '''
        depth = np.array(depth_img) / 255.0
        mask = np.array(mask_img)
        depth = (depth * (depth_max - depth_min) + depth_min) / unit

        # convert to float32
        depth = depth.astype(np.float32)

        # sample resize
        if resize:
            depth, mask = self.sample_resize(depth, mask, n)
        
        # back projection
        pc_xyz = self.back_projection(depth, unit=1., align_corners=align_corners)
        pts = self.mask_sample(pc_xyz, mask, n, sample_strategy=sample_strategy)

        # restrict the return type
        pts = pts.astype(np.float32)
        return pts

#nn.Module 
# TODO: implement this Module
class DepthToPC(nn.Module):
    def __init__(self, device, img_w=224, img_h=224, focal_div_sensor=35./32.):
        self.device = device
        self.img_w = img_w
        self.img_h = img_h
        self.u_0 = img_w / 2
        self.v_0 = img_h / 2
        self.f_u = img_w * focal_div_sensor
        self.f_v = img_h * focal_div_sensor

        self.grid_x, self.grid_y = np.mgrid[:self.img_w, :self.img_h]
        self.grid_x = nn.Parameter(self.grid_x, requires_grad=False)
        self.grid_y = nn.Parameter(self.grid_y, requires_grad=False)

    def forward(self, depth, unit=1.):
        # depth : B * 1 * w * h
        return depth
        
