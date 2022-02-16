import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
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

    def __init__(self, model, optimizer, device=None, input_type='img', vis_dir=None, pred_minmax=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.pred_minmax = pred_minmax

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
        eval_dict = {}

        with torch.no_grad():
            loss = self.compute_loss(data)

        eval_dict['eval_loss'] = loss.item()
        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        inputs = data.get('inputs').to(device)
        gt_depth_maps = data.get('inputs.depth')
        gt_masks = data.get('inputs.mask').byte()
        batch_size = gt_depth_maps.size(0)
        
        kwargs = {}
        self.model.eval()
        with torch.no_grad():
            pr_depth_maps = self.model(None, inputs, func='predict_depth_map').cpu()
        
        for i in trange(batch_size):
            gt_depth_map = gt_depth_maps[i]
            pr_depth_map = pr_depth_maps[i]
            gt_mask = gt_masks[i]
            
            pr_depth_map = depth_to_L(pr_depth_map, gt_mask)
            gt_depth_map = depth_to_L(gt_depth_map, gt_mask)

            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            input_depth_path = os.path.join(self.vis_dir, '%03d_in_depth.png' % i)
            pr_depth_path = os.path.join(self.vis_dir, '%03d_pr_depth.png' % i)
            vis.visualize_data(inputs[i].cpu(), 'img', input_img_path)
            vis.visualize_data(gt_depth_map, 'img', input_depth_path)
            vis.visualize_data(pr_depth_map, 'img', pr_depth_path)

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        inputs = data.get('inputs').to(device)
        gt_depth_maps = data.get('inputs.depth').to(device)
        gt_mask = data.get('inputs.mask').to(device)
        pr_depth_maps = self.model(None, inputs, func='predict_depth_maps')
        n_predicts = pr_depth_maps.size(1)

        mask_pix_count = gt_mask.sum()

        # TODO: make compute loss functions inside model.forward to balance gpu usage during DP forward
        loss = 0
        for i in range(n_predicts):
            # for object
            if mask_pix_count != 0.:
                loss += (F.mse_loss(pr_depth_maps[:,i], gt_depth_maps, reduce=False) * gt_mask).sum() / mask_pix_count
            # for background
            #loss += ( F.relu(5.0 - pr_depth_maps[:,i]) * (1. - gt_mask) ).mean()
            #loss += 0.1 * ( F.sigmoid(pr_depth_maps[:,i]) * (-1.0) * (1. - gt_mask) ).mean()

        if self.pred_minmax:
            gt_min = data.get('inputs.depth_min').to(device)
            gt_max = data.get('inputs.depth_max').to(device)
            predicted_minmax = self.model(None, None, func='fetch_minmax')
            w_minmax = (224 ** 2) / 2
            loss += w_minmax * F.mse_loss(predicted_minmax[:,0], gt_min)
            loss += w_minmax * F.mse_loss(predicted_minmax[:,1], gt_max)

        return loss