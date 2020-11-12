import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockConv1d, ResnetBlockFC
from im2mesh.encoder.pointnet import PointNetEncoder
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from im2mesh.common import make_2d_grid
from im2mesh.point_completion.models.model import PointDecoder

#TODO: Implement a PCN style network

class PCNFolding(nn.Module):
    '''
        PCN style point cloud completion network
        
    '''
    def __init__(self, encoder, device=None, 
        c_dim=1024, input_points_count=2048, skeleton_points_count=512,
        grid_scale=0.5, grid_size=3):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)

        self.grid_scale = grid_scale
        self.grid_size = grid_size

        self.skeleton_points_count = skeleton_points_count
        self.output_points_count = skeleton_points_count * (grid_size ** 2) #default = 4608

        self.coarse_decoder = PointDecoder(c_dim,output_points_count=skeleton_points_count)

    def build_2d_grid(self, batch_size):
        grid = make_2d_grid(
            [- self.grid_scale] * 2,
            [self.grid_scale] * 2,
            [self.grid_size ] * 2
        ).cuda()
        grid = grid.reshape(1, self.grid_size ** 2, 2).repeat(batch_size, self.skeleton_points_count, 1).contiguous()

        return grid # batch_size * (skeleton_points_count * grid_size ** 2) * 2

    def forward(self, x, world_mat=None):
        pointnet_encoder = False
        feats = self.encoder(x)
        
        if isinstance(feats, tuple):
            if len(feats) == 3:
                feats, _, trans_feature = feats
                pointnet_encoder = True

        coarse = self.coarse_decoder(feats)

        return 

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model