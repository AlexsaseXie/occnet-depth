import torch
import torch.nn as nn
from torch import distributions as dist

class SALNetwork(nn.Module):
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

    def forward(self, p, inputs=None, func='forward',
        gt_sal=None, z_loss_ratio=1.0e-3, z=None,
        **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        
        if func == 'forward':
            return self.forward_predict(p, inputs, **kwargs)
        elif func == 'infer_z':
            return self.infer_z(inputs, **kwargs)
        elif func == 'decode':
            return self.decode(p, z, **kwargs)
        elif func == 'z_loss':
            assert gt_sal is not None
            return self.z_loss(p, z, gt_sal, z_loss_ratio=z_loss_ratio, **kwargs)
        elif func == 'forward_loss':
            assert gt_sal is not None
            return self.forward_loss(p, inputs, gt_sal, z_loss_ratio=z_loss_ratio, **kwargs)

    def forward_predict(self, p, inputs, **kwargs):
        q_z, _= self.infer_z(inputs, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, **kwargs)

        return p_r

    def z_loss(self, p, z, gt_sal, z_loss_ratio=1.0e-3, sal_loss_type='l1', **kwargs):
        z_reg = z.abs().mean(dim=-1)
        p_r = self.decode(p, z, **kwargs)

        # sal loss
        if sal_loss_type == 'l1':
            loss_sal = torch.abs(p_r.abs() - gt_sal).mean()
        elif sal_loss_type == 'l2':
            loss_sal = torch.pow(p_r.abs() - gt_sal, 2).mean()
        else:
            raise NotImplementedError

        # latent loss: regularization
        loss_sal += z_loss_ratio * z_reg.mean()

        return loss_sal, p_r

    def forward_loss(self, p, inputs, gt_sal, z_loss_ratio=1.0e-3, sal_loss_type='l1', **kwargs):
        q_z, z_reg = self.infer_z(inputs, **kwargs)
        z = q_z.rsample()
        p_r = self.decode(p, z, **kwargs)

        # sal loss
        if sal_loss_type == 'l1':
            loss_sal = torch.abs(p_r.abs() - gt_sal).mean()
        elif sal_loss_type == 'l2':
            loss_sal = torch.pow(p_r.abs() - gt_sal, 2).mean()
        else:
            raise NotImplementedError

        # latent loss: regularization
        loss_sal += z_loss_ratio * z_reg.mean()

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
        ''' P
        uts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model