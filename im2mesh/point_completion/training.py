import os
from tqdm import trange
import torch
from torch.nn import functional as F
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer

from im2mesh.encoder.pointnet import feature_transform_reguliarzer, PointNetEncoder, PointNetResEncoder
from im2mesh.onet_depth.training import compose_inputs
from im2mesh.common import chamfer_distance, get_camera_mat, transform_points, project_to_camera, get_world_mat
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.lib_pointcloud_distance import emd


def compose_pointcloud(data, device, pointcloud_transfer=None):
    gt_pc = data.get('pointcloud').to(device)

    if pointcloud_transfer == 'world_scale_model':
        batch_size = gt_pc.size(0)
        gt_pc_loc = data.get('pointcloud.loc').to(device).view(batch_size, 1, 3)
        gt_pc_scale = data.get('pointcloud.scale').to(device).view(batch_size, 1, 1)

        gt_pc = gt_pc * gt_pc_scale + gt_pc_loc

    return gt_pc
    

class PointCompletionTrainer(BaseTrainer):
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

    def __init__(self, model, optimizer, device=None, input_type='depth_pointcloud',
                 vis_dir=None, 
                 depth_pointcloud_transfer='world_scale_model',
                 gt_pointcloud_transfer='world_scale_model',
                 view_penalty=False,
                 loss_type='cd'
                ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        assert input_type == 'depth_pointcloud'
        self.vis_dir = vis_dir
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        self.depth_pointcloud_transfer = depth_pointcloud_transfer
        assert depth_pointcloud_transfer in (None, 'world', 'world_scale_model', 'transpose_xy')

        self.gt_pointcloud_transfer = gt_pointcloud_transfer
        assert gt_pointcloud_transfer in (None, 'world_scale_model')

        self.mesh_evaluator = MeshEvaluator() 
        self.view_penalty = view_penalty
        self.loss_type = loss_type

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

        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer)
        batch_size = gt_pc.size(0)
        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer)

        with torch.no_grad():
            world_mat = None
            if self.model.encoder_world_mat is not None:
                if 'world_mat' in raw_data:
                    world_mat = raw_data['world_mat']
                else:
                    world_mat = get_world_mat(data, device=device)
                out = self.model(encoder_inputs, world_mat = world_mat)
            else:
                out = self.model(encoder_inputs)

            if isinstance(out, tuple):
                out, trans_feat = out

                #if isinstance(self.model.encoder, PointNetEncoder) or isinstance(self.model.encoder, PointNetResEncoder):
                #    loss = loss + 0.001 * feature_transform_reguliarzer(trans_feat)

            eval_dict = {}
            if batch_size == 1:
                pointcloud_hat = out.cpu().squeeze(0).numpy()
                pointcloud_gt = gt_pc.cpu().squeeze(0).numpy()
            
                eval_dict = self.mesh_evaluator.eval_pointcloud(pointcloud_hat, pointcloud_gt)

            # chamfer distance loss
            if self.loss_type == 'cd':
                loss = chamfer_distance(out, gt_pc).mean()
                eval_dict['chamfer'] = loss.item()
            else:
                out_pts_count = out.size(1)
                loss = (emd.earth_mover_distance(out, gt_pc, transpose=False) / out_pts_count).mean()
                eval_dict['emd'] = loss.item()
            

            # view penalty loss
            if self.view_penalty:
                gt_mask = data.get('inputs.mask').to(device) # B * 1 * H * W
                if world_mat is None:
                    world_mat = get_world_mat(data, device=device)
                camera_mat = get_camera_mat(data, device=device)

                # projection use world mat & camera mat
                out_pts = transform_points(out, world_mat)
                out_pts_img = project_to_camera(out_pts, camera_mat)
                out_pts_img = out_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

                out_mask = F.grid_sample(gt_mask, out_pts_img) # B * 1 * 1 * n_pts
                # B * n_pts
                
                loss_mask = (1. - out_mask).sum(dim=3).mean()
                loss_mask = 0.00001 * loss_mask

                eval_dict['view_penalty'] = loss_mask.item()

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer)
        batch_size = gt_pc.size(0)
        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)
        self.model.eval()

        with torch.no_grad():
            world_mat = None
            if self.model.encoder_world_mat is not None:
                if 'world_mat' in raw_data:
                    world_mat = raw_data['world_mat']
                else:
                    world_mat = get_world_mat(data, device=device)
                out = self.model(encoder_inputs, world_mat=world_mat)
            else:
                out = self.model(encoder_inputs)
        
        if isinstance(out, tuple):
            out, _ = out        


        for i in trange(batch_size):
            pc = gt_pc[i].cpu()
            vis.visualize_pointcloud(pc, out_file=os.path.join(self.vis_dir, '%03d_gt_pc.png' % i))

            pc = out[i].cpu()
            vis.visualize_pointcloud(pc, out_file=os.path.join(self.vis_dir, '%03d_pr_pc.png' % i))

            pc = encoder_inputs[i].cpu()
            vis.visualize_pointcloud(pc, out_file=os.path.join(self.vis_dir, '%03d_input_half_pc.png' % i))

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer)

        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)

        loss = 0

        world_mat = None
        if self.model.encoder_world_mat is not None:
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
            out = self.model(encoder_inputs, world_mat = world_mat)
        else:
            out = self.model(encoder_inputs)

        if isinstance(out, tuple):
            out, trans_feat = out

            if isinstance(self.model.encoder, PointNetEncoder) or isinstance(self.model.encoder, PointNetResEncoder):
                loss = loss + 0.001 * feature_transform_reguliarzer(trans_feat) 

        # chamfer distance loss
        if self.loss_type == 'cd':
            loss = loss + chamfer_distance(out, gt_pc).mean()
        else:
            out_pts_count = out.size(1)
            loss = loss + (emd.earth_mover_distance(out, gt_pc, transpose=False) / out_pts_count).mean()

        # view penalty loss
        if self.view_penalty:
            gt_mask = data.get('inputs.mask').to(device) # B * 1 * H * W
            if world_mat is None:
                world_mat = get_world_mat(data, device=device)
            camera_mat = get_camera_mat(data, device=device)

            # projection use world mat & camera mat
            out_pts = transform_points(out, world_mat)
            out_pts_img = project_to_camera(out_pts, camera_mat)
            out_pts_img = out_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

            out_mask = F.grid_sample(gt_mask, out_pts_img) # B * 1 * 1 * n_pts
            # B * n_pts
            
            loss_mask = (1. - out_mask).sum(dim=3).mean()
            loss = loss + 0.00001 * loss_mask
            
        return loss
