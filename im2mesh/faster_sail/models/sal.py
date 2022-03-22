from turtle import forward
import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F
import numpy as np


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

class SimplePointnet_VAE(nn.Module):
    ''' PointNet-based encoder network. Based on: https://github.com/autonomousvision/occupancy_networks

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        torch.nn.init.constant_(self.fc_mean.weight,0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):

        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean,c_std

class Decoder(nn.Module):
    '''  Based on: https://github.com/facebookresearch/DeepSDF
    '''
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        activation=None,
        latent_dropout=False,
        initial_radius=1,
    ):
        super().__init__()

        print('Initial Radius for decoder:', initial_radius)

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3
            lin = nn.Linear(dims[l], out_dim)

            if (l in dropout):
                p = 1 - dropout_prob
            else:
                p = 1.0

            if l == self.num_layers - 2:
                # bug! unexpect 2 in 2 * np.sqrt(np.pi), now removed
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(p * dims[l]), std=0.000001)
                torch.nn.init.constant_(lin.bias, -initial_radius)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(p*out_dim))

            if weight_norm and l in self.norm_layers:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.use_activation = not activation is None

        if self.use_activation:
            self.last_activation = get_class(activation)()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout

    def forward(self, xyz, latent_vecs=None, func='batch'):
        if func == 'one':
            return self.forward_one(xyz, latent_vecs=latent_vecs)
        # xyz: B * n * 3
        # latent_vecs: B * z_dim
        if latent_vecs is not None:
            if latent_vecs.dim() == 2:
                latent_vecs = latent_vecs.expand(latent_vecs.size(0), xyz.size(1), latent_vecs.size(1))
            if self.latent_dropout:
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            raw_input = torch.cat([latent_vecs, xyz], 2)
            x = raw_input
        else:
            raw_input = xyz
            x = raw_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, raw_input], 2) /np.sqrt(2)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 2) /np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_activation:
            x = self.last_activation(x) + 1.0 * x

        x = x.squeeze(2)
        return x

    def forward_one(self, xyz, latent_vecs=None):
        # xyz: n * 3
        # latent_vecs: z_dim
        if latent_vecs is not None:
            if latent_vecs.dim() == 1:
                latent_vecs = latent_vecs.expand(xyz.size(0), latent_vecs.size(0))
            if self.latent_dropout:
                latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = xyz

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1) /np.sqrt(2)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1) /np.sqrt(2)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.use_activation:
            x = self.last_activation(x) + 1.0 * x

        x = x.squeeze(1)
        return x

encoder_latent_dict = {
    'pointnet_vae': SimplePointnet_VAE
}

decoder_dict = {
    'deepsdf': Decoder
}

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
        if z is not None:
            z_reg = z.abs().mean(dim=-1)
        else:
            z_reg = None
        p_r = self.decode(p, z, **kwargs)

        # sal loss
        if sal_loss_type == 'l1':
            loss_sal = torch.abs(p_r.abs() - gt_sal).mean()
        elif sal_loss_type == 'l2':
            loss_sal = torch.pow(p_r.abs() - gt_sal, 2).mean()
        else:
            raise NotImplementedError

        # latent loss: regularization
        if z_loss_ratio != 0 and z_reg is not None:
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
        if z_loss_ratio != 0:
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
        out = self.decoder(p, z, func='batch', **kwargs)
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