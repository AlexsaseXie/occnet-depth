import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class WorldMatEncoder(nn.Module):
    r''' Simple MLP Network to encode world mat

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=1024):
        super().__init__()
        self.fc_1 = nn.Linear(12, 64)
        self.fc_2 = nn.Linear(64, 256)
        self.fc_3 = nn.Linear(256, 512)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.LeakyReLU()

    def forward(self, world_mat):
        batch_size = world_mat.size(0)

        net = world_mat.reshape(batch_size, -1)

        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_3(net))
        out = self.fc_out(self.actvn(net))

        return out