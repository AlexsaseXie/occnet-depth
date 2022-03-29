import os
from threading import local
from tqdm import trange, tqdm
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
from im2mesh.faster_sail.utils import find_r_t, pcwrite

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

                data_z = torch.randn(batch_size, self.optim_z_dim) * 1e-3
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
            z_learning_rate=1e-3, device=None, vis_dir=None,
            K=128, initial_length_alpha=1.5, initial_z_std=1e-3, random_subfield=0):
        self.model = model
        self.optimizer = optimizer

        self.device = device
        self.vis_dir = vis_dir

        self.optim_z_dim = optim_z_dim # int or [int]
        assert self.optim_z_dim > 0
        self.z_learning_rate = z_learning_rate
        self.z_optimizer = None

        self.point_range = None
        self.point_sample = None
        self.surface_point_weight = 0

        self.K = K
        self.initial_length_alpha = initial_length_alpha
        self.initial_z_std = initial_z_std
        print('K: %d, initial length alpha: %f, z_std: %f' % (self.K, self.initial_length_alpha, self.initial_z_std))
        print('Initial z learning rate:', self.z_learning_rate)

        self.random_subfield = random_subfield
        self.surface_point_weight = 0
        
        # status
        self.cube_set_K = CubeSet(device, refine_center=False, refine_length=False)
        self.training_p = None
        self.training_calc_index = None

    def init_z(self, data, initialize_optimizer=True):
        device = self.device
        z_vec = data.get('z', None)
        if z_vec is not None:
            print('Init with existing params')
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

                print('Conduct fps to get initial %d centers...' % K)
                pointcloud_idx = pointnet2_utils.furthest_point_sample(pointcloud, K) 
                pointcloud_K = pointnet2_utils.gather_operation(pointcloud_flipped, pointcloud_idx).transpose(1, 2).contiguous()
                # pointcloud_K: B * K * 3

                X = pointcloud_K.unsqueeze(2).repeat(1,1,K,1)
                Y = pointcloud_K.unsqueeze(1).repeat(1,K,1,1)

                #dis = torch.sqrt(((X - Y) ** 2).sum(dim=3)) # B * K * K, Dis[i][j] = distance between i and j
                dis = (X - Y).abs().max(dim=3)[0] # B * K * K, Dis[i][j] = PI_x distance between i and j
                dis, _ = dis.topk(7, dim=2, largest=False) # B * K * 7

                initial_length = dis[:,:,1:].mean(dim=2) * (self.initial_length_alpha / 2.0) # B * K
                z_vec = (torch.randn((B, K, self.optim_z_dim)) * self.initial_z_std).to(device)

                # initial_length_cpu = initial_length.cpu()
                # pointcloud_K_cpu = pointcloud_K.cpu()
                # del initial_length, pointcloud_K, pointcloud_idx

                self.cube_set_K.set(pointcloud_K, initial_length, z_vec)
                # set r t
                self.init_K_neighbor_r_t(data, pointcloud_K, initial_length)
                print('Set %d cube set' % K)
                print('Length\'s avg:', self.cube_set_K.length.mean())

        if initialize_optimizer:
            param_list = self.cube_set_K.learnable_parameters()
            print('Learnable params:', param_list)
            #self.z_optimizer = optim.SGD([ param_list[k] for k in param_list ], lr=self.z_learning_rate)
            self.z_optimizer = optim.Adam([ param_list[k] for k in param_list ], lr=self.z_learning_rate)

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

    def init_pointcloud_points_record(self, data):
        device = self.device

        self.training_pc = data.get('inputs.pointcloud').to(device)
        self.training_pc_calc_index = self.cube_set_K.query(self.training_pc)
        print('Training pc calc index:', self.training_pc_calc_index.shape)

    def init_training_points(self, p):
        '''
            p: B * N * 3 points already on device
        '''
        with torch.no_grad():
            self.training_calc_index = self.cube_set_K.query(p)
            print('Training calc index:', self.training_calc_index.shape)

            if self.random_subfield != 0:
                self.training_calc_index_sep, total_len = self.cube_set_K.query_sep(p)
                print('Training calc index sep: batch %d, K %d, subfield 0 shape:' % 
                    (len(self.training_calc_index_sep), 
                    len(self.training_calc_index_sep[0])), 
                    self.training_calc_index_sep[0][0].shape, 
                    ',Total len: %d' % total_len
                )

    def show_points(self, output_dir):
        print('Show points')
        from im2mesh.utils.lib_pointcloud_voxel import grid_points_query_range
        with torch.no_grad():
            centers = self.cube_set_K.center.cpu().numpy()[0,:,:]
            output_file = os.path.join(output_dir, 'centers.ply')
            pcwrite(output_file, centers, color=False)
            output_file = os.path.join(output_dir, 'center5.ply')
            pcwrite(output_file, centers[10:15,:], color=False)

            t = 5
            output_file = os.path.join(output_dir, 'batch_points.ply')
            lst = []
            neighbors = []
            for i in range(10,10+t):
                batch_data = self.cube_set_K.get(self.training_p, self.training_calc_index_sep[0][i])
                cur_points = batch_data['input_p'].cpu().numpy()
                cur_color = np.ones((cur_points.shape[0], 3), dtype=np.float32) * 256 * (i-10) / t
                cur_xyzrgb = np.concatenate([cur_points, cur_color], axis=1)
                lst.append(cur_xyzrgb)
                neighbors.append(self.neighbor_points[i])
            
            xyzrgb = np.concatenate(lst, axis=0)
            pcwrite(output_file, xyzrgb, color=True)

            neighbors = np.concatenate(neighbors, axis=0)
            output_file = os.path.join(output_dir, 'neighbor_pc.ply')
            pcwrite(output_file, neighbors, color=False)  

            pc_numpy = self.training_pc.cpu().numpy()[0]
            voxel, calc_index, inside_index, outside_index = grid_points_query_range(
                pc_numpy, 
                256, 0.005, 
                -0.55, 0.55
            )

            need_calc_points = voxel[calc_index[:,0], calc_index[:,1], calc_index[:,2], :3]
            output_file = os.path.join(output_dir, '256_voxel_need_calc_pc.ply')
            pcwrite(output_file, need_calc_points, color=False)

            need_calc_points_torch = torch.from_numpy(need_calc_points).unsqueeze(0)
            vs = self.predict_for_points_fast(need_calc_points_torch).cpu().numpy()
            blank_ids = (vs == 0)
            blank_points = need_calc_points[blank_ids]

            output_file = os.path.join(output_dir, '256_voxel_blank_pc.ply')
            pcwrite(output_file, blank_points, color=False)

    def init_K_neighbor_r_t(self, data, pointcloud_K, initial_length):
        assert pointcloud_K.shape[0] == 1
        pointcloud_K_numpy = pointcloud_K.cpu().numpy().squeeze(0)
        initial_length_numpy = initial_length.cpu().numpy().squeeze(0)
        pc_numpy = data.get('inputs.pointcloud').numpy().squeeze(0)

        device = self.device

        r_s = []
        t_s = []
        self.neighbor_points = []
        for i in range(self.K):
            center = pointcloud_K_numpy[i,:]
            initial_length = initial_length_numpy[i]

            local_pc = pc_numpy - center
            dis_to_center = np.abs(local_pc).max(axis=1)
            neighbors = pc_numpy[dis_to_center < initial_length]
            self.neighbor_points.append(neighbors)
            local_pc = local_pc[dis_to_center < initial_length] / initial_length

            assert local_pc.shape[0] > 0
            r, t = find_r_t(local_pc)
            r_s.append(r)
            t_s.append(t)

        r_s = np.array(r_s)[np.newaxis, :, np.newaxis].astype(np.float32) # 1 * n * 1
        t_s = np.array(t_s)[np.newaxis, :].astype(np.float32) # 1 * n * 3 

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

    def predict_for_points(self, p, batch_size=1000000, hj=None, weight_func='sail_s3_paper'):
        device = self.device
        B = p.shape[0]
        assert B == 1
        n_points = p.shape[1]
        with torch.no_grad():
            p = p.to(device)
            
            calc_index = self.cube_set_K.query(p)

            value = torch.zeros(n_points, dtype=torch.float32)
            weight = torch.zeros(n_points, dtype=torch.float32)
            calc_index_s = torch.split(calc_index, batch_size)

            print('p:', p.shape)
            print('Calc index:', calc_index.shape)
            for cs in calc_index_s:
                batch_data = self.cube_set_K.get(p, calc_index=cs)
                unified_weight = batch_data['unified_weight']
                p_r = self.model(batch_data['input_unified_coordinate'], 
                    z=batch_data['input_z'], func='decode', 
                    unified_weight=unified_weight
                ).cpu()

                for i in range(cs.shape[0]):
                    b_i = cs[i,0]
                    p_i = cs[i,1]
                    center_i = cs[i,2]

                    cur_p = batch_data['input_p'][i]
                    center = self.cube_set_K.center[b_i, center_i]
                    length = self.cube_set_K.length[b_i, center_i]

                    if weight_func == 'sail_s3_paper':
                        w = ((cur_p - center).abs().max() - length).abs().cpu()
                    elif weight_func == 'uniform_far':
                        w = ((cur_p - center).abs().max() / length)
                        w = (1.0 / w).cpu()
                    elif weight_func == 'length_far':
                        w = ((cur_p - center).abs().max() - length).abs()
                        w = (1.0 / w).cpu()
                    else:
                        raise NotImplementedError

                    weight[p_i] += w
                    if hj is None:
                        value[p_i] += p_r[i] * w # * h(j)
                    else:
                        value[p_i] += p_r[i] * w * hj[center_i]

            p_r = value / weight

        return p_r

    def predict_for_points_fast(self, p, batch_size=1000000, hj=None, weight_func='sail_s3_paper', aggregate='mean'):
        device = self.device
        B = p.shape[0]
        assert B == 1
        n_points = p.shape[1]
        with torch.no_grad():
            p = p.to(device)
            
            calc_index = self.cube_set_K.query(p)
            calc_index_s = torch.split(calc_index, batch_size)

            print('Fast predict... p:', p.shape)
            print('Calc index:', calc_index.shape)

            p_ids = calc_index[:,1].cpu().numpy().astype(np.float32)
            weights = []
            values = []
            for cs in calc_index_s:
                batch_data = self.cube_set_K.get(p, calc_index=cs)
                unified_weight = batch_data['unified_weight']
                p_r = self.model(batch_data['input_unified_coordinate'], 
                    z=batch_data['input_z'], func='decode', 
                    unified_weight=unified_weight
                )

                p_coors = batch_data['input_p']
                center_coors = self.cube_set_K.center[0, cs[:,2],:]
                lengths = self.cube_set_K.length[0, cs[:,2]]
                if hj is not None:
                    hjs = hj[cs[:,2]]
                else:
                    hjs = 1.
                
                if weight_func == 'sail_s3_paper':
                    ws = ((p_coors - center_coors).abs().max(dim=1)[0] - lengths).abs()
                elif weight_func == 'uniform_far':
                    ws = ((p_coors - center_coors).abs().max(dim=1)[0] / lengths)
                    ws = 1.0 / ws
                elif weight_func == 'length_far':
                    ws = ((p_coors - center_coors).abs().max(dim=1)[0] - lengths).abs()
                    ws = 1.0 / ws 
                else:
                    raise NotImplementedError
                ws = ws.cpu().numpy()
                p_r = p_r.cpu().numpy()
                vs = ws * hjs * p_r

                weights.append(ws)
                values.append(vs)

            weights = np.concatenate(weights, axis=0)
            values = np.concatenate(values, axis=0)
            infos = np.concatenate([p_ids.reshape(-1,1), weights.reshape(-1,1), values.reshape(-1,1)], axis=1) # T * 3

            infos = infos[infos[:, 0].argsort()]
            infos = np.split(infos, np.unique(infos[:, 0], return_index=True)[1][1:])
            if aggregate == 'mean':
                infos = np.array([ [tmp[0,0], tmp[:,1].sum(), tmp[:,2].sum()] for tmp in infos])
                infos[:,2] = infos[:,2] / infos[:,1]
            elif aggregate == 'max':
                infos = np.array([ [tmp[0,0], tmp[np.argmax(tmp[:,1]), 1], tmp[np.argmax(tmp[:,1]), 2] ] for tmp in infos])
                infos[:,2] = infos[:,2] / infos[:,1]
            else:
                raise NotImplementedError

            if infos.shape[0] != n_points:
                print('%d/%d eval points are not in any subfields.' % (n_points-infos.shape[0], n_points))
                results = np.zeros(n_points, dtype=np.float32)
                idx = infos[:,0].astype(np.int)
                results[idx] = infos[:,2]

                results = torch.from_numpy(results)
                return results
            else:
                results = torch.from_numpy(infos[:,2])
                return results

    def predict_for_points_with_specific_subfield(self, p, subfield_id, batch_size=1000000):
        device = self.device
        B = p.shape[0]
        assert B == 1
        n_points = p.shape[1]
        with torch.no_grad():
            p = p.to(device)
            
            b_id = torch.zeros((n_points, 1),dtype=torch.int64)
            p_id = torch.from_numpy(np.arange(n_points, dtype=np.int)).reshape(n_points, 1)
            center_id = torch.empty((n_points, 1), dtype=torch.int64)
            center_id[:] = subfield_id
            cs = torch.cat([b_id, p_id, center_id], dim=1)

            results = []
            calc_index_s = torch.split(cs, batch_size)
            for cs in calc_index_s:
                batch_data = self.cube_set_K.get(p, calc_index=cs)
                unified_weight = batch_data['unified_weight']
                p_r = self.model(batch_data['input_unified_coordinate'], z=batch_data['input_z'], func='decode', unified_weight=unified_weight)
                
                results.append(p_r)
            results = torch.cat(results, dim=0)

        return results

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
            if self.random_subfield == 0:
                rand_idx = np.random.choice(calc_index.shape[0], size=self.point_sample, replace=False)
                cs = calc_index[rand_idx,:]
            else:
                K = self.K
                B = p.shape[0]
                calc_index_list = []
                for b in range(B):
                    rand_subfield_idx = np.random.choice(K, size=self.random_subfield, replace=False)
                    for subfield_id in rand_subfield_idx:
                        cur_subfield = self.training_calc_index_sep[b][subfield_id]
                        total_len = cur_subfield.shape[0]
                        rand_idx = np.random.choice(total_len, size=self.point_sample // self.random_subfield)
                        calc_index_list.append(cur_subfield[rand_idx,:])
                cs = torch.cat(calc_index_list, dim=0)
            batch_data = self.cube_set_K.get(p, calc_index=cs, gt_sal_val=gt_sal_val)

            weight = batch_data['unified_weight']
            # print('Training input data')
            # for key in batch_data:
            #     print('%s:' % key, batch_data[key].shape, 'Dtype: %s' % batch_data[key].dtype,
            #          'Device: %s' % batch_data[key].device)
            loss, _ = self.model(batch_data['input_unified_coordinate'], gt_sal=batch_data['gt_sal'] / weight, 
                z=batch_data['input_z'], func='loss', sal_weight=weight)

            if self.surface_point_weight != 0:
                rand_idx = np.random.choice(self.training_pc_calc_index.shape[0], size=self.point_sample, replace=False)
                cs = self.training_pc_calc_index[rand_idx, :]
                
                batch_data = self.cube_set_K.get(self.training_pc, cs, gt_sal_val=None)

                weight = batch_data['unified_weight']
                loss_pc, _ = self.model(batch_data['input_unified_coordinate'], gt_sal=0, 
                    z=batch_data['input_z'], func='loss', sal_weight=weight)

                loss += loss_pc * self.surface_point_weight
        else:
            with torch.no_grad():
                calc_index_b = torch.split(calc_index, 500000)
                losses = 0
                total_size = calc_index.shape[0]
                for cs in tqdm(calc_index_b):
                    batch_size = cs.shape[0]
                    batch_data = self.cube_set_K.get(p, calc_index=cs, gt_sal_val=gt_sal_val)

                    weight = batch_data['unified_weight']
                    cur_loss, _ = self.model(batch_data['input_unified_coordinate'], gt_sal=batch_data['gt_sal'] / weight, 
                        z=batch_data['input_z'], func='loss', sal_weight=weight)

                    losses += cur_loss * batch_size

                loss = losses / total_size

                # surface points
                if self.surface_point_weight != 0:
                    calc_index_b = torch.split(self.training_pc_calc_index, 500000)
                    losses_pc = 0
                    total_size = self.training_pc_calc_index.shape[0]
                    for cs in tqdm(calc_index_b):
                        batch_size = cs.shape[0]
                        batch_data = self.cube_set_K.get(p, calc_index=cs, gt_sal_val=None)

                        weight = batch_data['unified_weight']
                        cur_loss, _ = self.model(batch_data['input_unified_coordinate'], gt_sal=0, 
                            z=batch_data['input_z'], func='loss', sal_weight=weight)

                        losses_pc += cur_loss * batch_size

                    losses_pc = losses_pc / total_size
                    
                    loss += losses_pc * self.surface_point_weight
        
        return loss
        