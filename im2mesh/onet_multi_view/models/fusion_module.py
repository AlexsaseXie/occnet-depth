import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
    ResnetBlockConv1d
)

class FusionModule(nn.Module):
    ''' FusionModule class.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
    '''

    def __init__(self, dim=3, c_dim=128, n_views = 3,
                 hidden_size=128, leaky=False):
        super().__init__()
        assert c_dim > 0
        self.c_dim = c_dim
        self.n_views = n_views

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)
        self.c_block0 = ResnetBlockFC(hidden_size)
        self.c_block1 = ResnetBlockFC(hidden_size)
        self.c_block2 = ResnetBlockFC(hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c=None, **kwargs):
        batch_size, T, D = p.size()
        batch_size, n_views, c_dim = c.size()

        net = self.fc_p(p)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)



        net_c = self.fc_c(c).unsqueeze(1)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out