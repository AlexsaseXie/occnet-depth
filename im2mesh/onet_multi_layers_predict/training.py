import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from im2mesh.common import (
    compute_iou, make_3d_grid, fix_K_camera, get_camera_args
)
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer

from im2mesh.onet.loss_functions import get_occ_loss, occ_loss_postprocess

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
        eval_sample (bool): whether to evaluate samples

    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5, eval_sample=False, loss_type='cross_entropy', 
                 use_local_feature=False, surface_loss_weight=1.,
                 binary_occ=False, 
                 loss_tolerance_episolon=0.,
                 sign_lambda=0.
                ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.loss_type = loss_type
        self.use_local_feature = use_local_feature
        self.surface_loss_weight = surface_loss_weight
        self.binary_occ = binary_occ
        self.loss_tolerance_episolon = loss_tolerance_episolon
        self.sign_lambda = sign_lambda

        if self.surface_loss_weight != 1.:
            print('Surface loss weight:', self.surface_loss_weight)

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
        threshold = self.threshold
        eval_dict = {}

        # Compute elbo
        points = data.get('points').to(device)
        if self.binary_occ:
            occ = (data.get('points.occ') >= 0.5).float().to(device)
        else:
            occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        if self.binary_occ:
            occ_iou = (data.get('points_iou.occ') >= 0.5).float().to(device)
        else:
            occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        if self.use_local_feature:
            camera_args = get_camera_args(data, 'points.loc', 'points.scale', device=device)
            Rt = camera_args['Rt']
            K = camera_args['K']

        with torch.no_grad():
            if self.use_local_feature:
                elbo, rec_error, kl = self.model.compute_elbo(
                    points, occ, inputs, Rt, K, **kwargs)
            else:
                elbo, rec_error, kl = self.model.compute_elbo(
                    points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
            if self.use_local_feature:
                p_out = self.model(points_iou, inputs, Rt, K,
                                sample=self.eval_sample, **kwargs)
            else:
                p_out = self.model(points_iou, inputs,
                                sample=self.eval_sample, **kwargs)

        occ_iou_np = (occ_iou >= 0.5).cpu().numpy()
        occ_iou_hat_np = (p_out.probs >= threshold).cpu().numpy()
        iou = compute_iou(occ_iou_np, occ_iou_hat_np).mean()
        eval_dict['iou'] = iou

        # Estimate voxel iou
        if voxels_occ is not None:
            voxels_occ = voxels_occ.to(device)
            points_voxels = make_3d_grid(
                (-0.5 + 1/64,) * 3, (0.5 - 1/64,) * 3, (32,) * 3)
            points_voxels = points_voxels.expand(
                batch_size, *points_voxels.size())
            points_voxels = points_voxels.to(device)
            with torch.no_grad():
                if self.use_local_feature:
                    p_out = self.model(points_voxels, inputs, Rt, K,
                                    sample=self.eval_sample, **kwargs)
                else:   
                    p_out = self.model(points_voxels, inputs,
                                    sample=self.eval_sample, **kwargs)

            voxels_occ_np = (voxels_occ >= 0.5).cpu().numpy()
            occ_hat_np = (p_out.probs >= threshold).cpu().numpy()
            iou_voxels = compute_iou(voxels_occ_np, occ_hat_np).mean()

            eval_dict['iou_voxels'] = iou_voxels

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        if self.use_local_feature:
            camera_args = get_camera_args(data, 'points.loc', 'points.scale', device=device)
            Rt = camera_args['Rt']
            K = camera_args['K']

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
            if self.use_local_feature:
                p_r = self.model(p, inputs, Rt, K, sample=self.eval_sample, **kwargs)
            else:
                p_r = self.model(p, inputs, sample=self.eval_sample, **kwargs)

        occ_hat = p_r.probs.view(batch_size, *shape)
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        p = data.get('points').to(device)
        if self.binary_occ:
            occ = (data.get('points.occ') >= 0.5).float().to(device)
        else:
            occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)
        kwargs = {}

        if self.use_local_feature:
            camera_args = get_camera_args(data, 'points.loc', 'points.scale', device=device)
            Rt = camera_args['Rt']
            K = camera_args['K']
            f3,f2,f1 = self.model.encode_inputs(inputs,p,Rt,K)
        else:
            f3,f2,f1 = self.model.encode_inputs(inputs)
        
        q_z = self.model.infer_z(p, occ, f3, **kwargs)
        z = q_z.rsample()

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = kl.mean()

        # General points
        p_r = self.model.decode(p, z, f3, f2, f1, **kwargs)
        logits = p_r.logits
        probs = p_r.probs
        
        # loss
        loss_i = get_occ_loss(logits, occ, self.loss_type)
        # loss strategies
        loss_i = occ_loss_postprocess(loss_i, occ, probs, self.loss_tolerance_episolon, self.sign_lambda, self.threshold, self.surface_loss_weight)

        loss = loss + loss_i.sum(-1).mean()

        return loss
