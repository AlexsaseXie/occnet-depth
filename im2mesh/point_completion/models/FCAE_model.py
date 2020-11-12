import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockConv1d, ResnetBlockFC
from im2mesh.encoder.pointnet import PointNetEncoder
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils


class PointDecoder(nn.Module):
    def __init__(self, c_dim=1024, output_points_count=2048, hidden_size=1024):
        super(PointDecoder, self).__init__()
        self.fc_1 = ResnetBlockFC(c_dim, hidden_size)
        self.fc_2 = ResnetBlockFC(hidden_size, output_points_count)
        self.fc_out = nn.Linear(output_points_count, output_points_count * 3)

        self.output_points_count = output_points_count

    def forward(self, x):
        # x: feature B * c_dim
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_out(x)

        x = x.reshape(-1, self.output_points_count, 3)
        return x

'''
TODO: refine inputs
class PointRefineDecoder(nn.Module):
'''    

class PointCompletionNetwork(nn.Module):
    ''' PointCompletionNetwork Network class.

    Args:
        encoder:
        c_dim:
        device:
    '''

    def __init__(self, encoder, device=None, c_dim=1024, input_points_count=2048, output_points_count=2048,
     preserve_input=False, encoder_world_mat=None):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)

        self.output_points_count = output_points_count
        assert output_points_count % 2 == 0

        self.preserve_input = preserve_input

        if not self.preserve_input:
            self.decoder = PointDecoder(c_dim=c_dim, output_points_count=output_points_count).to(device)
        else:
            self.half_p = output_points_count // 2
            self.decoder = PointDecoder(c_dim=c_dim, output_points_count=self.half_p).to(device)
            # TODO: define a shortcut function

        if encoder_world_mat is not None:
            self.encoder_world_mat = encoder_world_mat.to(device)
        else:
            self.encoder_world_mat = None
            

    def forward(self, x, world_mat=None):
        pointnet_encoder = False
        feats = self.encoder(x)

        if isinstance(feats, tuple):
            feats, _, trans_feature = feats
            pointnet_encoder = True

        if self.encoder_world_mat is not None:
            feat_world_mat = self.encoder_world_mat(world_mat)
            # TODO: try other functions to incorporate world_mat's feat
            feats = feats + feat_world_mat

        points_output = self.decoder(feats)

        if self.preserve_input:
            # TODO: make shortcut function work
            x_flipped = x.transpose(1,2).contiguous()
            points_transferred_idx = pointnet2_utils.furthest_point_sample(x, self.half_p)
            points_transferred = pointnet2_utils.gather_operation(x_flipped, points_transferred_idx).transpose(1, 2).contiguous()
            points_output = torch.cat([points_output, points_transferred], dim=1)
        
        if pointnet_encoder:
            return points_output, trans_feature
        else:
            return points_output

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
