
from im2mesh.onet.models.decoder import Decoder, DecoderBatchNorm, DecoderCBatchNorm, DecoderCBatchNorm2, DecoderCBatchNormNoResnet

import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)


class DecoderCBatchNorm3(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim 
        p_hidden_size = 256
        hidden_size = 128
       
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, p_hidden_size)

        self.fc_p = nn.Conv1d(dim, p_hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, p_hidden_size, size_out = hidden_size * 4, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size * 4 + p_hidden_size, size_out= hidden_size * 4, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size * 4 + p_hidden_size, size_out= hidden_size * 2, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size * 2 + p_hidden_size, size_out= hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net_initial = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net_initial = net_initial + net_z

        net = self.block0(net_initial, c)
        net = torch.cat((net, net_initial),dim=1)
        net = self.block1(net, c)
        net = torch.cat((net, net_initial),dim=1)
        net = self.block2(net, c)
        net = torch.cat((net, net_initial),dim=1)
        net = self.block3(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out

class DecoderBatchNormConcat(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert self.c_dim != 0
        self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        
        self.block0 = ResnetBlockConv1d(hidden_size * 2)
        self.block1 = ResnetBlockConv1d(hidden_size * 2)
        self.block2 = ResnetBlockConv1d(hidden_size * 2)
        self.block3 = ResnetBlockConv1d(hidden_size * 2)
        self.block4 = ResnetBlockConv1d(hidden_size * 2, size_out=hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c).unsqueeze(2).repeat(1,1,T)
        net = torch.cat((net, net_c),dim=1)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out

class DecoderConcat(nn.Module):
    ''' Decoder with batch normalization class.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=128, c_dim=128,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert self.c_dim != 0
        self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        
        self.block0 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block1 = nn.Conv1d(hidden_size * 2, hidden_size, 1)
        self.block2 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block3 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block4 = nn.Conv1d(hidden_size * 2, hidden_size, 1)


        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c).unsqueeze(2).repeat(1,1,T)
        net = torch.cat((net, net_c),dim=1)
        net = self.block0(net)
        net = self.block1(net)

        net = torch.cat((net, net_c),dim=1)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out

class DecoderBatchNorm_LocalFeature(nn.Module):
    ''' Decoder with batch normalization class using local feature c.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=64,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert c_dim != 0
        assert hidden_size % 2 == 0
        self.fc_c = nn.Conv1d(c_dim, hidden_size // 2, 1)
        self.fc_p = nn.Conv1d(dim, hidden_size // 2, 1)
        self.block0 = ResnetBlockConv1d(hidden_size, size_out=hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)
        self.block3 = ResnetBlockConv1d(hidden_size)
        self.block4 = ResnetBlockConv1d(hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        c = c.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c)
        net = torch.cat((net, net_c), dim=1)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out

class DecoderBatchNormSimple_LocalFeature(nn.Module):
    ''' Decoder with batch normalization class using local feature c.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=64,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert c_dim != 0
        assert hidden_size % 2 == 0
        self.fc_c = nn.Conv1d(c_dim, hidden_size // 2, 1)
        self.fc_p = nn.Conv1d(dim, hidden_size // 2, 1)
        self.block0 = ResnetBlockConv1d(hidden_size, size_out=hidden_size)
        self.block1 = ResnetBlockConv1d(hidden_size)
        self.block2 = ResnetBlockConv1d(hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        c = c.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c)
        net = torch.cat((net, net_c), dim=1)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out

class Decoder_LocalFeature(nn.Module):
    ''' Decoder with batch normalization class using local feature c.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=64,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert c_dim != 0
        assert hidden_size % 2 == 0
        self.fc_c = nn.Conv1d(c_dim, hidden_size, 1)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block1 = nn.Conv1d(hidden_size * 2, hidden_size, 1)
        self.block2 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block3 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block4 = nn.Conv1d(hidden_size * 2, hidden_size, 1)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        c = c.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c)
        net = torch.cat((net, net_c), dim=1)
        net = self.block0(net)
        net = self.block1(net)

        net = torch.cat((net, net_c), dim=1)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out

class DecoderSimple_LocalFeature(nn.Module):
    ''' Decoder with batch normalization class using local feature c.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=64,
                 hidden_size=256, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert c_dim != 0
        assert hidden_size % 2 == 0
        self.fc_c = nn.Conv1d(c_dim, hidden_size, 1)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block1 = nn.Conv1d(hidden_size * 2, hidden_size * 2, 1)
        self.block2 = nn.Conv1d(hidden_size * 2, hidden_size * 1, 1)

        self.norm = nn.InstanceNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        c = c.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c)
        net = torch.cat((net, net_c), dim=1)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)

        out = self.fc_out(self.actvn(self.norm(net)))
        out = out.squeeze(1)

        return out

class DecoderBatchNormHighHidden_LocalFeature(nn.Module):
    ''' Decoder with batch normalization class using local feature c.
        While hidden layers dimension varies.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, z_dim=0, c_dim=256,
                  hidden_size=128, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        hidden_size=128
        # Submodules
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        assert c_dim != 0
        assert hidden_size % 2 == 0
        self.fc_c = nn.Conv1d(c_dim, hidden_size, 1)
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = ResnetBlockConv1d(hidden_size * 2, size_out=hidden_size * 4)
        self.block1 = ResnetBlockConv1d(hidden_size * 4 + hidden_size * 2, size_out=hidden_size * 4)
        self.block2 = ResnetBlockConv1d(hidden_size * 4 + hidden_size * 2, size_out=hidden_size * 2)
        self.block3 = ResnetBlockConv1d(hidden_size * 2 + hidden_size * 2, size_out=hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = nn.ReLU(inplace=True)
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2, True)

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        c = c.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        net_c = self.fc_c(c)
        net_initial = torch.cat((net, net_c), dim=1)

        net = self.block0(net_initial)
        net = torch.cat((net, net_initial),dim=1)
        net = self.block1(net)
        net = torch.cat((net, net_initial),dim=1)
        net = self.block2(net)
        net = torch.cat((net, net_initial),dim=1)
        net = self.block3(net)

        out = self.fc_out(self.actvn(self.bn(net)))
        out = out.squeeze(1)

        return out
