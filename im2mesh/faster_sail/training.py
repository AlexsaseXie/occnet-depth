import os
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
        if optim_z_dim != 0 and optim_z_dim is not None:
            self.z_device = None
            self.z_learning_rate = z_learning_rate
            self.z_optimizer = None

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

        data_z = data.get('z', None)
        if data_z is None:
            p = data.get('p')
            batch_size = p.size(0)
            point_size = p.size(1)

            data_z = torch.randn(batch_size, self.optim_z_dim)
        else:
            # following code should never be used
            assert data_z.size(1) == self.optim_z_dim
        #self.z_device = torch.tensor(data_z, device=device, requires_grad=True)
        self.z_device = data_z.requires_grad_().to(device)
        self.z_optimizer = optim.SGD([self.z_device], lr=self.z_learning_rate)

    def train_step(self, data, steps=1):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
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
        self.clear_z()
        return loss.item()

    def eval_step(self, data):
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
            refine_step = 50
            self.init_z(data)
            for i in range(refine_step):
                self.z_optimizer.zero_grad()
                loss = self.compute_loss(data)
                loss.backward()
                self.z_optimizer.step()
            self.clear_z()

        eval_dict['eval_loss'] = loss.item()
        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        #TODO
        pass

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        gt_sal_val = data.get('points.sal').to(device)

        if self.with_encoder:
            inputs = data.get('inputs').to(device)
            loss = self.model(p, inputs=inputs, func='loss',
                gt_sal=gt_sal_val, z_loss_ratio=1.0e-3)
        else:
            loss = self.model(p, func='z_loss',
                gt_sal=gt_sal_val, z_loss_ratio=1.0e-3, z=self.z_device)
            
        return loss