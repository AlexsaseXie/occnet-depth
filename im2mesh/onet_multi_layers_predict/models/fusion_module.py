import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)
import torch

class FusionModule(nn.Module):
    ''' FusionModule class.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, c_dim=128, n_views = 3, pt_size = 2048,
                 hidden_size=128, leaky=False):
        super().__init__()
        assert c_dim > 0
        self.c_dim = c_dim
        self.n_views = n_views
        self.pt_size = pt_size

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)

        self.conv0 = nn.Sequential(
            nn.Conv1d(pt_size, 1024, 1),
            nn.InstanceNorm1d(1024),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.InstanceNorm1d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(512, 1, 1),
            nn.InstanceNorm1d(1),
            nn.ReLU()
        )
        
        self.fc_c = nn.Linear(c_dim, hidden_size)
        self.c_block0 = ResnetBlockFC(hidden_size)
        self.c_block1 = ResnetBlockFC(hidden_size)
        self.c_block2 = ResnetBlockFC(hidden_size)

        self.predict_actv = nn.ReLU()
        self.predict_fc0 = nn.Linear(2 * hidden_size, 1024)
        self.predict_fc1 = nn.Linear(1024, pt_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c, logits, **kwargs):
        batch_size, T, D = p.size()
        batch_size, n_views, c_dim = c.size()

        # p : batch_size * pt_size * 3

        net = self.fc_p(p)
        net = self.block0(net)
        net = self.conv0(net)
        net = self.block1(net)
        net = self.conv1(net)
        net = self.block2(net)
        net = self.conv2(net)
        
        # net : batch_size * 1 * 128

        # c : batch_size * n_views * c_dim
        net_c = self.fc_c(c)
        net_c = self.c_block0(net_c)
        net_c = self.c_block1(net_c)
        net_c = self.c_block2(net_c)

        # net_c : batch_size * n_views * 128
        net = net.repeat((1,self.n_views,1))
        net = torch.cat((net,net_c), 1)

        net = self.predict_actv(self.predict_fc0(net))
        net = self.predict_fc1(net)
        net = F.softmax( net , dim=1 )
        # net : batch_size * n_views * pt_size
        # logits : batch_size * n_views * pt_size
        
        out = (logits * net).sum(1)

        return out