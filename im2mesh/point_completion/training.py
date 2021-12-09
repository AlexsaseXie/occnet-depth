import os
from tqdm import trange
import torch
from torch.nn import functional as F
from im2mesh.utils import visualize as vis
from im2mesh.training import BaseTrainer

from im2mesh.encoder.pointnet import feature_transform_reguliarzer, PointNetEncoder, PointNetResEncoder
from im2mesh.onet_depth.training import compose_inputs, organize_space_carver_kwargs
from im2mesh.common import get_camera_mat, transform_points, project_to_camera, get_world_mat
from im2mesh.eval import MeshEvaluator
from im2mesh.utils.lib_pointcloud_distance import emd, chamfer_distance as cd
from im2mesh.onet_depth.models import background_setting


def compose_pointcloud(data, device, pointcloud_transfer=None, world_mat=None):
    gt_pc = data.get('pointcloud').to(device)

    if pointcloud_transfer == 'world_random_scale':
        pointcloud_transfer = 'world_scale_model'

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

    return gt_pc, gt_pc_scale, gt_pc_loc
    
# TODO: calc view penalty loss during FCAE module forward
# in order to fit DP/DDP
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
                 mask_flow_eps=1e-2,
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
        assert depth_pointcloud_transfer in ('world', 'world_scale_model', 'view', 'view_scale_model', 'world_random_scale')

        self.gt_pointcloud_transfer = gt_pointcloud_transfer
        assert gt_pointcloud_transfer in ('world_normalized', 'world_scale_model', 'view', 'view_scale_model', 'world_random_scale')

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
            self.mask_flow_eps = mask_flow_eps
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
        gt_pc, _, _ = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)
        batch_size = gt_pc.size(0)

        with torch.no_grad():
            if self.model.encoder_world_mat is not None:
                loss, out = self.model(encoder_inputs, world_mat=world_mat, gt_pc=gt_pc, 
                    loss_type=self.loss_type, train_loss=False)
            else:
                loss, out = self.model(encoder_inputs, gt_pc=gt_pc, 
                    loss_type=self.loss_type, train_loss=False)

            eval_dict = {}
            if batch_size == 1:
                pointcloud_hat = out.cpu().squeeze(0).numpy()
                pointcloud_gt = gt_pc.cpu().squeeze(0).numpy()
            
                eval_dict = self.mesh_evaluator.eval_pointcloud(pointcloud_hat, pointcloud_gt)

            if self.gt_pointcloud_transfer in ('world_scale_model', 'view_scale_model', 'view'):
                pointcloud_scale = data.get('pointcloud.scale').to(device).view(batch_size, 1, 1)
                loss = loss / (pointcloud_scale ** 2)
                if self.gt_pointcloud_transfer == 'view':
                    if world_mat is None:
                        world_mat = get_world_mat(data, device=device)
                    t_scale = world_mat[:, 2:, 3:]
                    loss = loss * (t_scale ** 2)   
 
            loss = loss.mean()
            if self.loss_type == 'cd':
                eval_dict['chamfer'] = loss.item()
            else:
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
                loss_mask_flow = F.relu(1. - self.mask_flow_eps - out_mask_flow, inplace=True).mean()
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
                    eval_dict['loss_depth_test'] = loss.item()

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
        gt_pc, _, _ = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)
        batch_size = gt_pc.size(0)

        self.model.eval()
        with torch.no_grad():
            if self.model.encoder_world_mat is not None:
                out = self.model(encoder_inputs, world_mat=world_mat, train_loss=False)
            else:
                out = self.model(encoder_inputs, train_loss=False)
               
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
        gt_pc, gt_pc_scale, gt_pc_loc = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)

        if self.depth_pointcloud_transfer == 'world_random_scale':
            assert self.gt_pointcloud_transfer == 'world_random_scale'
            batch_size = gt_pc.size(0)
            model_random_scale = data.get('inputs.model_random_scale').to(device).view(batch_size, 1, 1)

            rescale = model_random_scale / gt_pc_scale

            encoder_inputs = encoder_inputs * rescale
            gt_pc = gt_pc * rescale

        if self.model.encoder_world_mat is not None:
            loss, out = self.model(encoder_inputs, world_mat=world_mat, gt_pc=gt_pc, 
                loss_type=self.loss_type, train_loss=True)
        else:
            loss, out = self.model(encoder_inputs, gt_pc=gt_pc, 
                loss_type=self.loss_type, train_loss=True)

        # Make sure loss is still working under DP
        loss = loss.mean()

        # view penalty loss
        if self.view_penalty:
            gt_mask_flow = data.get('inputs.mask_flow').to(device) # B * 1 * H * W
            if world_mat is None:
                world_mat = get_world_mat(data, device=device)
            camera_mat = get_camera_mat(data, device=device)

            # projection use world mat & camera mat
            if self.gt_pointcloud_transfer == 'world_scale_model':
                out_pts = transform_points(out, world_mat)
            elif self.gt_pointcloud_transfer == 'world_random_scale':
                out_pts = out / rescale
                out_pts = transform_points(out_pts, world_mat)
            elif self.gt_pointcloud_transfer == 'view_scale_model':
                t = world_mat[:,:,3:]
                out_pts = out + t
            elif self.gt_pointcloud_transfer == 'view':
                t = world_mat[:,:,3:]
                out_pts = out * t[:,2:,:]
                out_pts = out_pts + t
            else:
                raise NotImplementedError
            
            out_pts_img = project_to_camera(out_pts, camera_mat)
            out_pts_img = out_pts_img.unsqueeze(1) # B * 1 * n_pts * 2

            out_mask_flow = F.grid_sample(gt_mask_flow, out_pts_img) # B * 1 * 1 * n_pts
            loss_mask_flow = F.relu(1. - self.mask_flow_eps - out_mask_flow, inplace=True).mean()
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


MSN_TRAINING_EMD_EPS=0.005
MSN_TRAINING_EMD_ITER=50
MSN_EVAL_EMD_EPS=0.002
MSN_EVAL_EMD_ITER=10000

class MSNTrainer(BaseTrainer):
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
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        self.depth_pointcloud_transfer = depth_pointcloud_transfer
        assert depth_pointcloud_transfer in ('world', 'world_scale_model', 'view', 'view_scale_model', 'world_random_scale')

        self.gt_pointcloud_transfer = gt_pointcloud_transfer
        assert gt_pointcloud_transfer in ('world_normalized', 'world_scale_model', 'view', 'view_scale_model', 'world_random_scale')

        if depth_pointcloud_transfer != gt_pointcloud_transfer:
            print('Warning: using different transfer for depth_pc & gt_pc.')
            print('Depth pc transfer: %s' % depth_pointcloud_transfer)
            print('Gt pc transfer: %s' % gt_pointcloud_transfer)
        else:
            print('Using %s for depth_pc & gt_pc' % depth_pointcloud_transfer)

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

        encoder_inputs, raw_data = compose_inputs(data, mode='train', device=self.device, input_type=self.input_type,
                                                depth_pointcloud_transfer=self.depth_pointcloud_transfer)
        world_mat = None
        if self.gt_pointcloud_transfer in ('view', 'view_scale_model'):
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
        gt_pc, _, _ = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)
        batch_size = gt_pc.size(0)

        kwargs = {}
        if getattr(self.model, 'module', False):
            space_carver_mode = getattr(self.model.module, 'space_carver_mode', False)
        else:
            space_carver_mode = getattr(self.model, 'space_carver_mode', False)
        if space_carver_mode:
            kwargs = organize_space_carver_kwargs(
                space_carver_mode, kwargs, 
                raw_data, data, device,
                target_space=self.gt_pointcloud_transfer
            )

        with torch.no_grad():
            _, out, _ = self.model(encoder_inputs, **kwargs)

            eval_dict = {}
            if batch_size == 1:
                pointcloud_hat = out.cpu().squeeze(0).numpy()
                pointcloud_gt = gt_pc.cpu().squeeze(0).numpy()
            
                eval_dict = self.mesh_evaluator.eval_pointcloud(pointcloud_hat, pointcloud_gt)

            loss = emd.earth_mover_distance(out, gt_pc, transpose=False)

            if self.gt_pointcloud_transfer in ('world_scale_model', 'view_scale_model', 'view'):
                pointcloud_scale = data.get('pointcloud.scale').to(device).view(batch_size, 1, 1)
                loss = loss / (pointcloud_scale ** 2)
                if self.gt_pointcloud_transfer == 'view':
                    if world_mat is None:
                        world_mat = get_world_mat(data, device=device)
                    t_scale = world_mat[:, 2:, 3:]
                    loss = loss * (t_scale ** 2)   

                out_pts_count = out.size(1)
                loss = (loss / out_pts_count).mean()
                eval_dict['emd'] = loss.item()
 
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
        if self.gt_pointcloud_transfer in ('view', 'view_scale_model'):
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
        gt_pc, _, _ = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)
        batch_size = gt_pc.size(0)

        kwargs = {}
        if getattr(self.model, 'module', False):
            space_carver_mode = getattr(self.model.module, 'space_carver_mode', False)
        else:
            space_carver_mode = getattr(self.model, 'space_carver_mode', False)
        if space_carver_mode:
            kwargs = organize_space_carver_kwargs(
                space_carver_mode, kwargs, 
                raw_data, data, device,
                target_space=self.gt_pointcloud_transfer
            )

        self.model.eval()
        with torch.no_grad():
            _, out, _ = self.model(encoder_inputs, **kwargs)
              
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
        if  self.gt_pointcloud_transfer in ('view', 'view_scale_model'):
            if 'world_mat' in raw_data:
                world_mat = raw_data['world_mat']
            else:
                world_mat = get_world_mat(data, device=device)
        gt_pc, gt_pc_scale, gt_pc_loc = compose_pointcloud(data, device, self.gt_pointcloud_transfer, world_mat=world_mat)

        if self.depth_pointcloud_transfer == 'world_random_scale':
            assert self.gt_pointcloud_transfer == 'world_random_scale'
            batch_size = gt_pc.size(0)
            model_random_scale = data.get('inputs.model_random_scale').to(device).view(batch_size, 1, 1)

            rescale = model_random_scale / gt_pc_scale

            encoder_inputs = encoder_inputs * rescale
            gt_pc = gt_pc * rescale

        # data parallel
        emd1, emd2, expansion_penalty = self.model(encoder_inputs, gt_pc=gt_pc, eps=MSN_TRAINING_EMD_EPS, it=MSN_TRAINING_EMD_ITER)

        loss = emd1.mean() + emd2.mean() + expansion_penalty.mean() * 0.1
        return loss
