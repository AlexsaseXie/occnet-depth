import os
from threading import local
from tqdm import trange
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch import device, distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer
import numpy as np
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from im2mesh.faster_sail.models.sail_s3 import CubeSet
from im2mesh.faster_sail.utils import find_r_t

class SALTrainer(BaseTrainer):
    ''' SALTrainer object for the SALNetwork.

    Args:
        model (nn.Module): SALNetwork model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        
    '''

    def __init__(self, model, optimizer, optim_z_dim=0, 
            z_learning_rate=1e-4, device=None, vis_dir=None, with_encoder=True):
        self.model = model
        self.optimizer = optimizer

        self.device = device
        self.vis_dir = vis_dir
        self.with_encoder = with_encoder

        self.optim_z_dim = optim_z_dim # int or [int]
        #if optim_z_dim != 0 and optim_z_dim is not None:
        self.z_device = None
        self.z_learning_rate = z_learning_rate
        self.z_optimizer = None
        self.point_range = None
        self.point_sample = None
        self.surface_point_weight = 0

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def clear_z(self):
        if self.optim_z_dim == 0:
            return
        self.z_device = None
        self.z_optimizer = None

    def init_z(self, data):
        if self.optim_z_dim == 0:
            return

        with torch.no_grad():
            data_z = data.get('z', None)
            if data_z is None:
                p = data.get('points')
                batch_size = p.size(0)
                point_size = p.size(1)

                data_z = torch.randn(batch_size, self.optim_z_dim)
            else:
                # following code should never be used
                assert data_z.size(1) == self.optim_z_dim
            #self.z_device = torch.tensor(data_z, device=device, requires_grad=True)
            self.z_device = data_z.requires_grad_().to(self.device).detach()
        self.z_optimizer = optim.SGD([self.z_device], lr=self.z_learning_rate)

    def train_step(self, data, steps=1, initialize_z=False):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        if initialize_z:
            self.init_z(data)
        self.model.train()
        for i in range(steps):
            self.optimizer.zero_grad()
            if self.z_optimizer is not None:
                self.z_optimizer.zero_grad()
            loss = self.compute_loss(data)
            loss.backward()
            self.optimizer.step()
            if self.z_optimizer is not None:
                self.z_optimizer.step()
        # TODO: may memorize z
        if initialize_z:
            self.clear_z()
        return loss.item()

    def eval_step(self, data, initialize_z=False, refine_step=50):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
        eval_dict = {}

        if self.optim_z_dim == 0:
            with torch.no_grad():
                loss = self.compute_loss(data)
        else:
            if initialize_z:
                self.init_z(data)
                for i in range(refine_step):
                    self.z_optimizer.zero_grad()
                    loss = self.compute_loss(data)
                    loss.backward()
                    self.z_optimizer.step()
                self.clear_z()
            else:
                with torch.no_grad():
                    loss = self.compute_loss(data)

        eval_dict['loss'] = loss.item()
        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        #TODO
        device = self.device
        batch_size = data['points'].size(0)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        self.model.eval()
        with torch.no_grad():
            if self.with_encoder:
                inputs = data.get('inputs').to(device)
                p_r = self.model(p, inputs=inputs, func='forward_predict')
            else:
                p_r = self.model(p, func='decode', z=self.z_device)
        
        sdf_hat = p_r.view(batch_size, *shape)
        voxels_out = (sdf_hat <= 0).cpu().numpy()

        for i in trange(batch_size):
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

        pass

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        if self.model.training and self.point_range is not None:
            assert len(self.point_range) == 2
            p = data.get('points')[:, self.point_range[0]:self.point_range[1]]
            gt_sal_val = data.get('points.sal')[:, self.point_range[0]:self.point_range[1]]
        else:
            p = data.get('points')
            gt_sal_val = data.get('points.sal')

        if self.model.training and self.point_sample is not None:
            rand_idx = np.random.choice(p.shape[1], size=self.point_sample, replace=False)
            p = p[:, rand_idx]
            gt_sal_val = gt_sal_val[:, rand_idx]
        p = p.to(device)
        gt_sal_val = gt_sal_val.to(device)

        if self.with_encoder:
            inputs = data.get('inputs.pointcloud').to(device)
            loss, _ = self.model(p, inputs=inputs, func='loss',
                gt_sal=gt_sal_val, z_loss_ratio=1.0e-3)
        else:
            loss, _ = self.model(p, func='z_loss',
                gt_sal=gt_sal_val, z_loss_ratio=1.0e-3, z=self.z_device)
            
        if self.surface_point_weight != 0:
            surface_points = data.get('inputs.pointcloud')
            if self.model.training and self.point_sample is not None:
                rand_idx = np.random.choice(surface_points.shape[1], size=self.point_sample, replace=False)
                surface_points = surface_points[:, rand_idx]
            surface_points = surface_points.to(device)

            if self.with_encoder:
                p_r = self.model(surface_points, inputs=inputs, func='forward_predict')
            else:
                p_r = self.model(surface_points, func='decode', z=self.z_device)

            loss_surface = p_r.abs().mean()
            loss += self.surface_point_weight * loss_surface
        
        return loss

class SAIL_S3_Trainer(BaseTrainer):
    def __init__(self, model, optimizer, optim_z_dim=0, 
            z_learning_rate=1e-4, device=None, vis_dir=None,
            K=128, initial_length_alpha=1.5, initial_z_std=1e-3):
        self.model = model
        self.optimizer = optimizer

        self.device = device
        self.vis_dir = vis_dir

        self.optim_z_dim = optim_z_dim # int or [int]
        self.z_learning_rate = z_learning_rate
        self.z_optimizer = None

        self.point_range = None
        self.point_sample = None
        self.surface_point_weight = 0

        self.K = K
        self.initial_length_alpha = initial_length_alpha
        self.initial_z_std = initial_z_std
        # status
        self.cube_set_K = CubeSet(refine_center=False, refine_length=False, device=device)
        self.training_p = None
        self.training_calc_index = None
        

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def init_z(self, data):
        device = self.device
        z_vec = data.get('z', None).to(device)
        if z_vec is not None:
            z_vec = z_vec.to(device)
            pointcloud_K = data.get('center').to(device)
            initial_length = data.get('length').to(device)
            r_s_tensor = data.get('r_tensor').to(device)
            t_s_tensor = data.get('t_tensor').to(device)

            self.cube_set_K.set(pointcloud_K, initial_length, z_vec)
            self.cube_set_K.set_initial_r_t(r_s_tensor, t_s_tensor)
        else:
            #initialize code
            with torch.no_grad():
                K = self.K
                # find K centers 
                pointcloud = data.get('inputs.pointcloud').to(device)
                pointcloud_flipped = pointcloud.transpose(1, 2).contiguous()

                B = pointcloud.shape[0]
                assert B == 1

                pointcloud_idx = pointnet2_utils.furthest_point_sample(pointcloud, K) 
                pointcloud_K = pointnet2_utils.gather_operation(pointcloud_flipped, pointcloud_idx).transpose(1, 2).contiguous()
                # pointcloud_K: B * K * 3

                A = pointcloud_K.unsqueeze(2).repeat(1,1,K,1)
                B = pointcloud_K.unsqueeze(1).repeat(1,K,1,1)

                dis = torch.sqrt(((A - B) ** 2).sum(dim=3)) # B * K * K, Dis[i][j] = distance between i and j
                dis, _ = dis.topk(5, dim=2, largest=False)

                initial_length = dis[:,1:].mean(dim=1)  * (self.initial_length_alpha / 2.0) # B * K
                z_vec = torch.randn(B, K, self.optim_z_dim) * self.initial_z_std

                self.cube_set_K.set(pointcloud_K, initial_length, z_vec)
                # set r t
                self.init_K_neighbor_r_t(data, pointcloud_K, initial_length)
                self.init_training_points_record(data)

        param_list = self.cube_set_K.learnable_parameters()
        self.z_optimizer = optim.SGD(param_list, lr=self.z_learning_rate)

    def clear_z(self):
        self.z_optimizer = None
        self.training_p = None
        self.training_calc_index = None
        self.cube_set_K.clear()

    def init_training_points_record(self, data):
        device = self.device

        if self.point_range is not None:
            assert len(self.point_range) == 2
            self.training_p = data.get('points')[:, self.point_range[0]:self.point_range[1]]
            self.training_gt_sal = data.get('points.sal')[:, self.point_range[0]:self.point_range[1]]
        else:
            self.training_p = data.get('points')
            self.training_gt_sal = data.get('points.sal')

        self.training_p = self.training_p.to(device)
        self.training_gt_sal = self.training_gt_sal.to(device)
        self.init_training_points(self.training_p)

    def init_training_points(self, p):
        '''
            p: B * N * 3 points already on device
        '''
        with torch.no_grad():
            self.training_calc_index = self.cube_set_K.query(p)

    def init_K_neighbor_r_t(self, data, pointcloud_K, initial_length):
        pointcloud_K_numpy = pointcloud_K.cpu().numpy()
        initial_length_numpy = initial_length.cpu().numpy()
        pc_numpy = data.get('inputs.pointcloud').numpy()

        device = self.device

        r_s = []
        t_s = []
        for i in range(self.K):
            center = pointcloud_K_numpy[i,:]
            initial_length = initial_length_numpy[i]

            local_pc = pc_numpy - center
            dis_to_center = (local_pc ** 2).sum(axis=0)
            
            local_pc = local_pc[dis_to_center < initial_length] / initial_length

            r, t = find_r_t(local_pc)
            r_s.append(r)
            t_s.append(t)

        r_s = np.concatenate(r_s, axis=0) # n * 1
        t_s = np.concatenate(t_s, axis=0) # n * 3 

        r_s_tensor = torch.from_numpy(r_s).to(device)
        t_s_tensor = torch.from_numpy(t_s).to(device)

        self.cube_set_K.set_initial_r_t(r_s_tensor, t_s_tensor)

    def train_step(self, data, steps=1):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        for i in range(steps):
            self.optimizer.zero_grad()
            if self.z_optimizer is not None:
                self.z_optimizer.zero_grad()
            loss = self.compute_loss(data)
            loss.backward()
            self.optimizer.step()
            if self.z_optimizer is not None:
                self.z_optimizer.step()
        return loss.item()

    def predict_for_points(self, p, batch_size=100000):
        device = self.device
        B = p.shape[0]
        n_points = p.shape[1]
        with torch.no_grad():
            p = p.to(device)
            
            calc_index = self.cube_set_K.query(p)

            value = torch.zeros(n_points)
            weight = torch.zeros(n_points)
            calc_index_s = torch.split(calc_index, batch_size)
            for cs in calc_index_s:
                batch_data = self.cube_set_K.get(p, calc_index=cs)
                p_r = self.model(batch_data['input_unified_coordinate'], z=batch_data['input_z'], func='decode')

                for i in range(cs.shape[0]):
                    b_i = cs[i,0]
                    p_i = cs[i,1]
                    center_i = cs[i,2]

                    p = batch_data['input_p'][i]
                    center = self.cube_set_K.center[b_i, center_i]
                    length = self.cube_set_K.length[b_i, center_i]

                    w = ((p - center).abs().max() - length).abs()

                    weight[b_i, p_i] += w
                    value[b_i, p_i] += p_r[i] * w # * h(j)

            p_r = value / weight

        return p_r

    def eval_step(self, data):
        self.model.eval()
        device = self.device
        eval_dict = {}

        loss = self.compute_loss(data)

        eval_dict['loss'] = loss.item()
        return eval_dict

    def compute_loss(self, data):
        device = self.device
        p = self.training_p 
        if p is None:
            self.init_training_points_record(data)
            p = self.training_p
        
        calc_index = self.training_calc_index # M * 3
        gt_sal_val = self.training_gt_sal

        M = calc_index.shape[0]
        assert self.point_sample is not None
        
        if self.model.training:
            rand_idx = np.random.choice(calc_index.shape[1], size=self.point_sample, replace=False)
            calc_index = calc_index[rand_idx,:]

            batch_data = self.cube_set_K.get(p, calc_index=calc_index, gt_sal=gt_sal_val)

            weight = batch_data['unified_weight']
            
            loss, _ = self.model(batch_data['input_unified_coordinate'], gt_sal=batch_data['gt_sal'] / weight, 
                z=batch_data['input_z'], func='loss', sal_weight=weight)
        else:
            with torch.no_grad():
                calc_index_b = torch.split(calc_index, 100000)
                losses = 0
                total_size = calc_index.shape[0]
                for cs in calc_index_b:
                    batch_size = cs.shape[0]
                    batch_data = self.cube_set_K.get(p, calc_index=cs, gt_sal=gt_sal_val)

                    weight = batch_data['unified_weight']
                    loss, _ = self.model(batch_data['input_unified_coordinate'], gt_sal=batch_data['gt_sal'] / weight, 
                        z=batch_data['input_z'], func='loss', sal_weight=weight)

                    losses += loss * batch_size

                loss = losses / total_size
        
        return loss
        