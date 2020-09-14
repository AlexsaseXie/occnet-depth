import torch
from torch import nn
import numpy as np
from PIL import Image

# for single numpy image
class DepthToPCNp:
    def __init__(self, focal_div_sensor=35./32.):
        self.focal_div_sensor = focal_div_sensor
        
        self.grid_x, self.grid_y = np.mgrid[:224, :224]

    def sample_resize(self, depth_img, mask_img, n=2048):
        '''
            depth_img : PIL Image
            mask_img : PIL Image

            returns depth: img_w * img_h numpy.ndarray
                    mask: img_w * img_h numpy.ndarray
        '''
        depth = np.array(depth_img)
        img_w = depth.shape[0]
        img_h = depth.shape[1]
        mask = np.array(mask_img)
        current_count = mask.sum()
        if current_count < n:
            factor = np.ceil(np.sqrt(n / current_count))
            depth = 1.0 / depth
            new_depth_img = Image.fromarray(depth)
            new_depth_img = new_depth_img.resize((img_w * factor, img_h * factor), Image.BILINEAR)
            new_mask_img = mask_img.resize((img_w * factor, img_h * factor))

            depth = 1.0 / np.array(new_depth_img)
            mask = np.array(new_mask_img)
            
            current_count = mask.sum()
            assert current_count >= n

        return depth, mask
            

    def back_projection(self, depth, unit=1.):
        '''
            depth : img_w * img_h np.ndarray
            returns pc_xyz: img_w * img_h * 3
        '''

        img_w = depth.shape[0]
        img_h = depth.shape[1]
        u_0 = img_w / 2
        v_0 = img_h / 2
        f_u = img_w * self.focal_div_sensor
        f_v = img_h * self.focal_div_sensor

        if img_w == 224 and img_h == 224:
            grid_x = self.grid_x
            grid_y = self.grid_y
        else:
            grid_x, grid_y = np.mgrid[:img_w, :img_h]

        pc_x = (grid_x - u_0) * depth / f_u
        pc_y = (grid_y - v_0) * depth / f_v
        pc_xyz = np.stack((pc_x, pc_y, depth - unit), axis=2)

        return pc_xyz

    def mask_sample(self, pc_xyz, mask, n=None):
        '''
            mask : img_w * img_h
            pc_xyz : img_w * img_h * 3

            returns pts : n * 3 points
        '''
        assert pc_xyz.shape[0] == mask.shape[0]
        assert pc_xyz.shape[1] == mask.shape[1]

        pts = pc_xyz[mask]
        if n is not None:
            assert pts.shape[0] >= n
            choice = np.random.sample(range(n), False)
            pts = pts[choice]
        
        return pts

    def work(self, depth_img, mask_img, n=2048, unit=1.):
        depth, mask = self.sample_resize(depth_img, mask_img, n)
        pc_xyz = self.back_projection(depth, unit)
        pts = self.mask_sample(pc_xyz, mask, n)
        return pts

#nn.Module
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
        