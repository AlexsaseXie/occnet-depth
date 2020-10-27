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
from im2mesh.onet_depth.models import background_setting


def compose_pointcloud(data, device, pointcloud_transfer=None, world_mat=None):
    gt_pc = data.get('pointcloud').to(device)

    # default : 'world_normalized'

    if pointcloud_transfer in ('world_scale_model', 'view', 'view_scale_model'):
        batch_size = gt_pc.size(0)
        gt_pc_loc = data.get('pointcloud.loc').to(device).view(batch_size, 1, 3)
        gt_pc_scale = data.get('pointcloud.scale').to(device).view(batch_size, 1, 1)

        gt_pc = gt_pc * gt_pc_scale + gt_pc_loc

    if pointcloud_transfer in ('view', 'view_scale_model'):
        assert world_mat is not None
        R = world_mat[:, :, :3]
        gt_pc = transform_points(gt_pc, R)

    if pointcloud_transfer in ('view'):
        assert world_mat is not None
        t = world_mat[:, :, 3:]
        gt_pc = gt_pc / t[:,2:,:]

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
                 loss_mask_flow_ratio=10,
                 loss_depth_test_ratio=5,
                 depth_test_eps=3e-3,
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
        assert depth_pointcloud_transfer in ('world', 'world_scale_model', 'view', 'view_scale_model')

        self.gt_pointcloud_transfer = gt_pointcloud_transfer
        assert gt_pointcloud_transfer in ('world_normalized', 'world_scale_model', 'view', 'view_scale_model')

        if depth_pointcloud_transfer != gt_pointcloud_transfer:
            print('Warning: using different transfer for depth_pc & gt_pc.')
            print('Depth pc transfer: %s' % depth_pointcloud_transfer)
            print('Gt pc transfer: %s' % gt_pointcloud_transfer)
        else:
            print('Using %s for depth_pc & gt_pc' % depth_pointcloud_transfer)

        self.mesh_evaluator = MeshEvaluator() 
        self.view_penalty = view_penalty
        if self.view_penalty:
            self.loss_mask_flow_ratio = loss_mask_flow_ratio
            if self.view_penalty == 'mask_flow_and_depth':
                self.loss_depth_test_ratio = loss_depth_test_ratio
                self.depth_test_eps = depth_test_eps

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

        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer)
        world_mat = None
        if (self.model.encoder_world_mat is not None) \
            or self.gt_pointcloud_transfer in ('view', 'view_scale_model'):
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)
        batch_size = gt_pc.size(0)

        with torch.no_grad():
            if self.model.encoder_world_mat is not None:
                out = self.model(encoder_inputs, world_mat = world_mat)
            else:
                out = self.model(encoder_inputs)

            if isinstance(out, tuple):
                out, trans_feat = out

            eval_dict = {}
            if batch_size == 1:
                pointcloud_hat = out.cpu().squeeze(0).numpy()
                pointcloud_gt = gt_pc.cpu().squeeze(0).numpy()
            
                eval_dict = self.mesh_evaluator.eval_pointcloud(pointcloud_hat, pointcloud_gt)

            # chamfer distance loss
            if self.loss_type == 'cd':
                loss = chamfer_distance(out, gt_pc)
            else:
                loss = emd.earth_mover_distance(out, gt_pc, transpose=False)

            if self.gt_pointcloud_transfer in ('world_scale_model', 'view_scale_model', 'view'):
                pointcloud_scale = data.get('pointcloud.scale').to(device).view(batch_size, 1, 1)
                loss = loss / (pointcloud_scale ** 2)
                if self.gt_pointcloud_transfer == 'view':
                    if world_mat is None:
                        world_mat = get_world_mat(data, device=device)
                    t_scale = world_mat[:, 2:, 3:]
                    loss = loss * (t_scale ** 2)   
 
            if self.loss_type == 'cd':
                loss = loss.mean()
                eval_dict['chamfer'] = loss.item()
            else:
                out_pts_count = out.size(1)
                loss = (loss / out_pts_count).mean()
                eval_dict['emd'] = loss.item()
            

            # view penalty loss
            if self.view_penalty:
                gt_mask_flow = data.get('inputs.mask_flow').to(device) # B * 1 * H * W
                if world_mat is None:
                    world_mat = get_world_mat(data, device=device)
                camera_mat = get_camera_mat(data, device=device)

                # projection use world mat & camera mat
                if self.gt_pointcloud_transfer == 'world_scale_model':
                    out_pts = transform_points(out, world_mat)
                elif self.gt_pointcloud_transfer == 'view_scale_model':
                    t = world_mat[:,:,3:]
                    out_pts = out_pts + t
                elif self.gt_pointcloud_transfer == 'view':
                    t = world_mat[:,:,3:]
                    out_pts = out_pts * t[:,2:,:]
                    out_pts = out_pts + t
                else:
                    raise NotImplementedError
                
                out_pts_img = project_to_camera(out_pts, camera_mat)
                out_pts_img = out_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

                out_mask_flow = F.grid_sample(gt_mask_flow, out_pts_img) # B * 1 * 1 * n_pts
                loss_mask_flow = F.relu(1. - out_mask_flow, inplace=True).mean()
                loss = self.loss_mask_flow_ratio * loss_mask_flow
                eval_dict['loss_mask_flow'] = loss.item()

                if self.view_penalty == 'mask_flow_and_depth':
                    # depth test loss
                    t_scale = world_mat[:, 2, 3].view(world_mat.size(0), 1, 1, 1)
                    gt_mask = data.get('inputs.mask').byte().to(device)
                    depth_pred = data.get('inputs.depth_pred').to(device) * t_scale

                    background_setting(depth_pred, gt_mask)
                    depth_z = out_pts[:,:,2:].transpose(1, 2)
                    corresponding_z = F.grid_sample(depth_pred, out_pts_img) # B * 1 * 1 * n_pts
                    corresponding_z = corresponding_z.squeeze(1)

                    # eps = 0.05
                    loss_depth_test = F.relu(depth_z - self.depth_test_eps - corresponding_z, inplace=True).mean()
                    loss = self.loss_depth_test_ratio * loss_depth_test
                    eval_dict['loss_depth_test'] = loss

        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)
        world_mat = None
        if (self.model.encoder_world_mat is not None) \
            or self.gt_pointcloud_transfer in ('view', 'view_scale_model'):
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)
        batch_size = gt_pc.size(0)

        self.model.eval()
        with torch.no_grad():
            if self.model.encoder_world_mat is not None:
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

        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)

        world_mat = None
        if (self.model.encoder_world_mat is not None) \
            or self.gt_pointcloud_transfer in ('view', 'view_scale_model'):
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)

        if self.model.encoder_world_mat is not None:
            out = self.model(encoder_inputs, world_mat = world_mat)
        else:
            out = self.model(encoder_inputs)

        loss = 0
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
            gt_mask_flow = data.get('inputs.mask_flow').to(device) # B * 1 * H * W
            if world_mat is None:
                world_mat = get_world_mat(data, device=device)
            camera_mat = get_camera_mat(data, device=device)

            # projection use world mat & camera mat
            if self.gt_pointcloud_transfer == 'world_scale_model':
                out_pts = transform_points(out, world_mat)
            elif self.gt_pointcloud_transfer == 'view_scale_model':
                t = world_mat[:,:,3:]
                out_pts = out_pts + t
            elif self.gt_pointcloud_transfer == 'view':
                t = world_mat[:,:,3:]
                out_pts = out_pts * t[:,2:,:]
                out_pts = out_pts + t
            else:
                raise NotImplementedError
            
            out_pts_img = project_to_camera(out_pts, camera_mat)
            out_pts_img = out_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

            out_mask_flow = F.grid_sample(gt_mask_flow, out_pts_img) # B * 1 * 1 * n_pts
            loss_mask_flow = F.relu(1. - out_mask_flow, inplace=True).mean()
            loss = loss + self.loss_mask_flow_ratio * loss_mask_flow

            if self.view_penalty == 'mask_flow_and_depth':
                # depth test loss
                t_scale = world_mat[:, 2, 3].view(world_mat.size(0),1, 1, 1)
                gt_mask = data.get('inputs.mask').byte().to(device)
                depth_pred = data.get('inputs.depth_pred').to(device) * t_scale # absolute depth from view
                background_setting(depth_pred, gt_mask)
                depth_z = out_pts[:,:,2:].transpose(1, 2)
                corresponding_z = F.grid_sample(depth_pred, out_pts_img) # B * 1 * 1 * n_pts
                corresponding_z = corresponding_z.squeeze(1)

                # eps
                loss_depth_test = F.relu(depth_z - self.depth_test_eps - corresponding_z, inplace=True).mean()
                loss = loss + self.loss_depth_test_ratio * loss_depth_test
                   
            
        return loss
