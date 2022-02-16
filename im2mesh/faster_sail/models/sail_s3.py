import torch
import torch.nn as nn
from torch import distributions as dist

class CubeSet:
    def __init__(self, refine_center=False, refine_length=False, device=None):
        self.refine_center = refine_center
        self.refine_length = refine_length
        self.refine_z_vector = True
        self.device = device

    def set(self, center_vector, length_vector, z_vectors):
        if self.refine_center:
            self.center = nn.Parameter(center_vector)
        else:
            self.center = center_vector

        if self.refine_length:
            self.length = nn.Parameter(length_vector)
        else:
            self.length = length_vector

        self.z_vectors = nn.parameter(z_vectors)

    def learnable_parameters(self):
        return_list = {}
        if self.refine_center:
            return_list['center'] = self.center

        if self.refine_length:
            return_list['length'] = self.length

        return_list['z_vectors'] = self.z_vectors
        return return_list

    def query(self, p):
        '''
            p: B * N * 3
            self.center: B * K * 3
            self.length: B * K
            self.z_vectors: B * K * z_dim
        '''
        B = p.size(0) # better B == 1
        N = p.size(1)
        K = self.center.size(1)
        with torch.no_grad():
            a = p.unsqueeze(2).repeat(1,1,K,1) # B * N * K * 3
            b = self.center.view(B,1,K,3)
            dis, _ = (a - b).max(axis=3) # B * N * K
            cmp = self.length.view(B,1,K)

            calc_index = torch.nonzero(dis < cmp) # M * 3

            del a,b,dis,cmp
        return calc_index

    def get(self, p, calc_index):
        # gradient pass through this function
        
        b_index = calc_index[:, 0] # M
        p_index = calc_index[:, 1] # M
        center_index = calc_index[:, 2] # M 

        # gather
        self.input_p = p[b_index, p_index] # M * 3
        self.input_z = self.z_vectors[b_index, center_index] # M * 3

        # unified coordinates
        center_vec = self.center[b_index, center_index] # M * 3
        bounding_length = self.length[b_index, center_index] # M

        self.input_unified_coordinate = (self.input_p - center_vec) / bounding_length.view(-1,1) # [-1, 1]

        data = {
            'input_p': self.input_p,    # M * 3
            'input_z': self.input_z,    # M * z_dim
            'input_unified_coordiante': self.input_unified_coordinate, # M * 3
        }

        return data
            


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
        gt_sal=None, z=None,
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
            return self.forward_loss(p, inputs, gt_sal, **kwargs)

    def forward_loss(self, p, z, gt_sal, sal_loss_type='l1', **kwargs):
        p_r = self.decode(p, z, **kwargs)

        # sal loss
        if sal_loss_type == 'l1':
            loss_sal = torch.abs(p_r.abs() - gt_sal).mean()
        elif sal_loss_type == 'l2':
            loss_sal = torch.pow(p_r.abs() - gt_sal, 2).mean()
        else:
            raise NotImplementedError

        # latent loss: regularization
        #loss_sal += z_loss_ratio * z_reg.mean()

        return loss_sal

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

    def decode(self, p, z, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        out = self.decoder(p, z, **kwargs)
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