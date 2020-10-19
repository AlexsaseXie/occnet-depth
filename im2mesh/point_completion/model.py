import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockConv1d, ResnetBlockFC
from im2mesh.encoder.pointnet import PointNetEncoder




class PointDecoder(nn.Module):
    def __init__(self, c_dim=1024, output_points_count=2048):
        super(PointDecoder, self).__init__()
        self.fc_1 = ResnetBlockFC(c_dim, 1024)
        self.fc_2 = ResnetBlockFC(1024, output_points_count)
        self.fc_out = nn.Linear(output_points_count, output_points_count * 3)

        self.output_points_count = output_points_count

    def forward(self, x):
        # x: feature B * c_dim
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_out(x)

        x = x.reshape(-1, self.output_points_count, 3)
        return x
        
class PointCompletionNetwork(nn.Module):
    ''' PointCompletionNetwork Network class.

    Args:
        encoder:
        c_dim:
        device:
    '''

    def __init__(self, encoder, device=None, c_dim=1024, input_points_count=2048, output_points_count=2048, preserve_input=False):
        super().__init__()
        self.device = device
        self.encoder = encoder.to(device)

        self.output_points_count = output_points_count
        assert output_points_count % 2 == 0

        self.preserve_input = preserve_input

        if not self.preserve_input:
            self.decoder = PointDecoder(c_dim=c_dim, output_points_count=output_points_count).to(device)
        else:
            raise NotImplementedError
            half_p = output_points_count // 2
            self.decoder = PointDecoder(c_dim=c_dim, output_points_count=half_p).to(device)
            # TODO: define a shortcut function
            

    def forward(self, x):
        feats, trans_points, trans_feature = self.encoder(x)
        points_output = self.decoder(feats)

        if self.preserve_input:
            raise NotImplementedError
            # TODO: make shortcut function work
            points_transferred = self.shortcut(x)
            points_output = torch.cat([points_output, points_transferred], dim=1)
        
        return points_output, trans_feature

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
