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
                 vis_dir=None, threshold=0.5, eval_sample=False, calc_feature_category_loss=False,
                 record_feature_category=True, attractive_p=1e-3, repulsive_p=1e-1,feature_k=1e-1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        self.calc_feature_category_loss = calc_feature_category_loss
        self.record_feature_category = record_feature_category
        self.attractive_p = attractive_p
        self.repulsive_p = repulsive_p
        self.feature_k = feature_k

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss,force_loss = self.compute_loss(data)
        if self.calc_feature_category_loss:
            force_loss.backward(retain_graph=True)
        loss.backward()
        self.optimizer.step()
        if self.calc_feature_category_loss:
            return loss.item(), force_loss.item()
        else:
            return loss.item(), force_loss

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
        occ = data.get('points.occ').to(device)

        inputs = data.get('inputs', torch.empty(points.size(0), 0)).to(device)
        voxels_occ = data.get('voxels')

        points_iou = data.get('points_iou').to(device)
        occ_iou = data.get('points_iou.occ').to(device)

        kwargs = {}

        with torch.no_grad():
            elbo, rec_error, kl = self.model.compute_elbo(
                points, occ, inputs, **kwargs)

        eval_dict['loss'] = -elbo.mean().item()
        eval_dict['rec_error'] = rec_error.mean().item()
        eval_dict['kl'] = kl.mean().item()

        # Compute iou
        batch_size = points.size(0)

        with torch.no_grad():
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
        device = self.device

        batch_size = data['points'].size(0)
        inputs = data.get('inputs', torch.empty(batch_size, 0)).to(device)

        shape = (32, 32, 32)
        p = make_3d_grid([-0.5] * 3, [0.5] * 3, shape).to(device)
        p = p.expand(batch_size, *p.size())

        kwargs = {}
        with torch.no_grad():
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
        occ = data.get('points.occ').to(device)
        inputs = data.get('inputs', torch.empty(p.size(0), 0)).to(device)

        kwargs = {}

        c = self.model.encode_inputs(inputs)
        # batch_size * points_count  
        q_z = self.model.infer_z(p, occ, c, **kwargs)
        z = q_z.rsample()

        #record c
        if self.record_feature_category:
            c_idx = data.get('category')  # batch_size tensor
            for batch_id, single_c in enumerate(c):
                self.model.category_centers[c_idx[batch_id],:] = self.model.category_centers[c_idx[batch_id],:] + single_c.detach()

        loss = 0
        force_loss = 0

        # Category loss
        if self.calc_feature_category_loss:
            c_idx = data.get('category')
            a_current_category_center = self.model.pre_category_centers[c_idx].detach().to(device) # batch_size * c_dim
            a_d = F.pairwise_distance(a_current_category_center, c, p=2) 

            #compensate
            a_c_d = F.relu(self.feature_k - a_d)
            compensate_loss = - ((self.repulsive_p * a_c_d * a_c_d / (2.0)).sum())

            a_d = F.relu(a_d - self.feature_k / 3.)

            #attractive
            # f_a(x) = alpha * x / k
            #attractive_loss = (self.attractive_p * a_d * a_d / (2.0 * self.feature_k)).sum()

            # f_a(x) = alpha * (x - k / 3) (x >= k / 3)
            attractive_loss = (self.attractive_p * a_d * a_d / (2.0)).sum()

            #repulsive
            # f_r(x) = alpha * k / x
            tmp_i = torch.LongTensor(range(self.model.category_count)).repeat(c_idx.shape[0],1)
            current_category_center = self.model.pre_category_centers[tmp_i].detach().to(device)
            current_c = c[:,None,:].repeat(1,self.model.category_count,1)
            d = torch.sqrt( torch.pow(current_category_center - current_c, 2).sum(2) )
            #repulsive_loss = (-self.repulsive_p * self.feature_k * torch.log(d + 1e-8)).sum()
            
            #f_r(x) = alpha * (k - x) (x <= k)
            d = F.relu(self.feature_k - d)
            repulsive_loss = (self.repulsive_p * d * d / 2.0).sum()
            
            repulsive_loss = repulsive_loss + compensate_loss
            force_loss = attractive_loss + repulsive_loss 
            #loss = loss + force_loss

        # KL-divergence
        kl = dist.kl_divergence(q_z, self.model.p0_z).sum(dim=-1)
        loss = loss + kl.mean()

        # General points
        logits = self.model.decode(p, z, c, **kwargs).logits
        if self.loss_type == 'cross_entropy':
            loss_i = F.binary_cross_entropy_with_logits(
                logits, occ, reduction='none')
        else:
            logits = F.sigmoid(logits)
            loss_i = torch.pow((logits - occ), 2)
        loss = loss + loss_i.sum(-1).mean()

        return loss, force_loss
