import torch
import torch.nn as nn
from torch import device, distributions as dist
from im2mesh.faster_sail.models.sal import Decoder_One

class CubeSet:
    def __init__(self, device, refine_center=False, refine_length=False):
        self.device = device
        self.refine_center = refine_center
        self.refine_length = refine_length
        self.refine_z_vector = True
        self.device = device

        self.center = None
        self.length = None
        self.circle_t_vector = None
        self.circle_r_vector = None
        self.z_vectors = None

    def export(self):
        assert self.z_vectors is not None
        assert self.center is not None
        assert self.length is not None
        assert self.circle_r_vector is not None
        assert self.circle_t_vector is not None

        with torch.no_grad():
            out_dict = {
                'z': self.z_vectors.cpu(),
                'center': self.center.cpu(),
                'length': self.length.cpu(),
                'r_tensor': self.circle_r_vector.cpu(),
                't_tensor': self.circle_t_vector.cpu()
            }
        return out_dict

    def clear(self):
        self.center = None
        self.length = None
        self.z_vectors = None
        self.circle_t_vector = None
        self.circle_r_vector = None

    def set(self, center_vector, length_vector, z_vectors):
        with torch.no_grad():
            if self.refine_center:
                self.center = center_vector.to(self.device).requires_grad_()
            else:
                self.center = center_vector.detach()

            if self.refine_length:
                self.length = length_vector.to(self.device).requires_grad_()
            else:
                self.length = length_vector.detach()

            self.z_vectors = z_vectors.to(self.device).requires_grad_()

    def set_initial_r_t(self, r_vector, t_vector):
        with torch.no_grad():
            self.circle_r_vector = r_vector.detach()
            self.circle_t_vector = t_vector.detach()

    def learnable_parameters(self):
        return_list = {}
        if self.refine_center:
            return_list['center'] = self.center

        if self.refine_length:
            return_list['length'] = self.length

        return_list['z'] = self.z_vectors
        return return_list

    def query(self, p):
        '''
        Input:
            p: B * N * 3
            self.center: B * K * 3
            self.length: B * K

        Output:
            calc_index: M * 3 [...[b_index, p_index, center_index]...] on cpu
        '''
        p = p.to(self.device)
        B = p.size(0) # better B == 1
        N = p.size(1)
        K = self.center.size(1)
        with torch.no_grad():
            a = p.unsqueeze(2).repeat(1,1,K,1) # B * N * K * 3
            b = self.center.to(self.device).view(B,1,K,3)
            dis, _ = (a - b).abs().max(axis=3) # B * N * K
            cmp = self.length.to(self.device).view(B,1,K)

            calc_index = torch.nonzero(dis < cmp) # M * 3

            del a,b,dis,cmp

        calc_index_cpu = calc_index.cpu()
        del calc_index
        return calc_index_cpu

    def query_sep(self, p):
        p = p.to(self.device) # B * N * 3
        B = p.size(0) # better B == 1
        N = p.size(1)
        K = self.center.size(1)
        
        return_list = []
        total_len = 0
        with torch.no_grad():
            centers = self.center.to(self.device)
            lens = self.length.to(self.device)
            for b in range(B):
                cur_batch_list = []
                for i in range(K):
                    cur_center = centers[b, i] # 3
                    cur_length = lens[b, i] # 1
                    cur_points = p[b,:,:] # n * 3
                    
                    dis, _ = (cur_points - cur_center).abs().max(axis=1) # n

                    cur_calc_index = torch.nonzero(dis < cur_length) # M * 3
                    del dis
                    cur_calc_index_cpu = cur_calc_index.cpu()
                    del cur_calc_index

                    M = cur_calc_index_cpu.shape[0]
                    b_id = torch.empty((M, 1), dtype=torch.int64)
                    b_id[:,:] = b
                    center_id = torch.empty((M, 1), dtype=torch.int64)
                    center_id[:,:] = i
                    
                    info = torch.cat([b_id, cur_calc_index_cpu, center_id], dim=1)
                    cur_batch_list.append(info)

                    total_len += M
                return_list.append(cur_batch_list)

        return return_list, total_len

    def get(self, p, calc_index, gt_sal_val=None):
        '''
            return tensors on device
        '''
        # gradient pass through this function
        batch_size = calc_index.shape[0]

        b_index = calc_index[:, 0] # M
        p_index = calc_index[:, 1] # M
        center_index = calc_index[:, 2] # M 

        # gather
        input_p = p[b_index, p_index].to(self.device) # M * 3
        if gt_sal_val is not None:
            gt_sal = gt_sal_val[b_index, p_index].to(self.device) # M * 3
        else:
            gt_sal = None
        input_z = self.z_vectors[b_index, center_index].to(self.device) # M * 3

        # unified coordinates
        center_vec = self.center[b_index, center_index].to(self.device) # M * 3
        bounding_length = self.length[b_index, center_index].to(self.device) # M
        r_vec = self.circle_r_vector[b_index, center_index].to(self.device)
        t_vec = self.circle_t_vector[b_index, center_index].to(self.device)

        input_unified_coordinate = (input_p - center_vec) / bounding_length.view(batch_size,1) # in range of [-1, 1]
        input_unified_coordinate = (input_unified_coordinate - t_vec) / r_vec.view(batch_size,1) # coordinates initially lied on a unit circle
        first_weight = r_vec.view(batch_size)
        unified_weight = bounding_length.view(batch_size) * first_weight

        data = {
            'input_p': input_p,    # M * 3
            'input_z': input_z,    # M * z_dim
            'gt_sal': gt_sal,    # M
            'unified_weight': unified_weight, # M
            'first_weight': first_weight,
            'input_unified_coordinate': input_unified_coordinate, # M * 3
        }
        # on device

        return data
            
decoder_dict = {
    'deepsdf': Decoder_One
}

class SAIL_S3Network(nn.Module):
    ''' SAL Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network for latent code z
        device (device): torch device
    '''

    def __init__(self, decoder, encoder_latent=None, device=None, z_dim=64):
        super().__init__()
        self.decoder = decoder.to(device)

        if encoder_latent is not None:
            self.encoder_latent = encoder_latent.to(device)
        else:
            self.encoder_latent = None

        self._device = device
        self.z_dim = z_dim

    def forward(self, p=None, inputs=None, func='forward',
        gt_sal=None, z=None, z_loss_ratio=1.0e-3,
        **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        if func == 'infer_z':
            return self.infer_z(inputs, **kwargs)
        elif func == 'decode':
            assert z is not None
            return self.decode(p, z, **kwargs)
        elif func == 'loss':
            assert gt_sal is not None
            return self.forward_loss(p, z, gt_sal, z_loss_ratio=z_loss_ratio,  **kwargs)
        elif func == 'signed_loss':
            assert gt_sal is not None
            return self.forward_signed_loss(p, z, gt_sal, **kwargs)

    def forward_loss(self, p, z, gt_sal, z_loss_ratio=1.0e-3, sal_loss_type='l1', sal_weight=None, **kwargs):
        if z is not None:
            z_reg = z.abs().mean(dim=-1)
        else:
            z_reg = None
        p_r = self.decode(p, z, **kwargs)

        # sal loss
        if sal_loss_type == 'l1':
            loss_sal = torch.abs(p_r.abs() - gt_sal)
        elif sal_loss_type == 'l2':
            loss_sal = torch.pow(p_r.abs() - gt_sal, 2)
        else:
            raise NotImplementedError

        if sal_weight is not None:
            loss_sal = loss_sal * sal_weight
        loss_sal = loss_sal.mean()

        # latent loss: regularization
        if z_loss_ratio != 0 and z_reg is not None:
            loss_sal += z_loss_ratio * z_reg.mean()

        return loss_sal, p_r

    def forward_signed_loss(self, p, z, gt_sal, sal_loss_type='l1', sal_weight=None, **kwargs):
        p_r = self.decode(p, z, **kwargs)

        # sal loss
        if sal_loss_type == 'l1':
            loss_sal = torch.abs(p_r - gt_sal)
        elif sal_loss_type == 'l2':
            loss_sal = torch.pow(p_r - gt_sal, 2)
        else:
            raise NotImplementedError

        if sal_weight is not None:
            loss_sal = loss_sal * sal_weight
        loss_sal = loss_sal.mean()

        return loss_sal, p_r

    def infer_z(self, inputs, **kwargs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder_latent is not None:
            mean_z, logstd_z = self.encoder_latent(inputs, **kwargs)
        else:
            batch_size = inputs.size(0)
            mean_z = torch.empty(batch_size, 0).to(self._device)
            logstd_z = torch.empty(batch_size, 0).to(self._device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        z_reg = mean_z.abs().mean(dim=-1) + logstd_z.abs().mean(dim=-1)

        return q_z, z_reg

    def decode(self, p, z, unified_weight=None, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        out = self.decoder(p, z, **kwargs)
        if unified_weight is not None:
            out = out * unified_weight
        return out

    def to(self, device):
        ''' 
        Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model