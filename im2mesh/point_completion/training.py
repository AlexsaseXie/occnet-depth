import os
from tqdm import trange
import torch
from torch.nn import functional as F
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer

from im2mesh.common import get_camera_args, get_world_mat, transform_points, transform_points_back
from im2mesh.encoder.pointnet import feature_transform_reguliarzer, PointNetEncoder, PointNetResEncoder
from im2mesh.onet_depth.training import compose_inputs
from im2mesh.common import chamfer_distance
from im2mesh.eval import MeshEvaluator

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
                ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        assert input_type == 'depth_pointcloud'
        self.vis_dir = vis_dir

        self.depth_pointcloud_transfer = depth_pointcloud_transfer
        assert depth_pointcloud_transfer in (None, 'world', 'world_scale_model', 'transpose_xy')

        self.gt_pointcloud_transfer = gt_pointcloud_transfer
        assert gt_pointcloud_transfer in (None, 'world_scale_model')

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        self.mesh_evaluator = MeshEvaluator() 

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
        encoder_inputs,_ = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)

        with torch.no_grad():
            loss = 0
            out = self.model(encoder_inputs)
            if isinstance(out, tuple):
                out, trans_feat = out

                #if isinstance(self.model.encoder, PointNetEncoder) or isinstance(self.model.encoder, PointNetResEncoder):
                #    loss = loss + 0.001 * feature_transform_reguliarzer(trans_feat)

            # chamfer distance loss
            loss = loss + chamfer_distance(out, gt_pc).mean()
            
            pointcloud_hat = out.cpu().squeeze(0).numpy()
            pointcloud_gt = gt_pc.cpu().squeeze(0).numpy()
            
            eval_dict = self.mesh_evaluator.eval_pointcloud(pointcloud_hat, pointcloud_gt)
            eval_dict['chamfer'] = loss.item()


        return eval_dict

    def visualize(self, data):
        ''' Performs a visualization step for the data.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        gt_pc = compose_pointcloud(data, device, self.gt_pointcloud_transfer)
        batch_size = gt_pc.size(0)
        encoder_inputs,_ = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)
        self.model.eval()
        with torch.no_grad():
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

        encoder_inputs,_ = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer,)

        loss = 0
        out = self.model(encoder_inputs)
        if isinstance(out, tuple):
            out, trans_feat = out

            if isinstance(self.model.encoder, PointNetEncoder) or isinstance(self.model.encoder, PointNetResEncoder):
                loss = loss + 0.001 * feature_transform_reguliarzer(trans_feat) 

        # chamfer distance loss
        loss = loss + chamfer_distance(out, gt_pc).mean()
        return loss
