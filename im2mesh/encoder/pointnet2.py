import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC, ResnetBlockConv1d
import numpy as np
import torch.nn.functional as F
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule, PointnetSAModuleMSG, build_shared_mlp
from im2mesh.utils.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import QueryAndGroup, ball_query, grouping_operation


class PointNet2SSGEncoder(nn.Module):
    def __init__(self, c_dim=1024, use_xyz=True, initial_feats_dim=0, local=False, local_feature_dim=512):
        super().__init__()

        self.c_dim = c_dim
        self.use_xyz = use_xyz
        self.local = local
        self.local_feature_dim = local_feature_dim
        self.initial_feats_dim = initial_feats_dim

        self._build_model()            

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[self.initial_feats_dim, 64, 64, 128],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, self.c_dim], use_xyz=self.use_xyz
            )
        )

        if self.local:
            self.xyz_fc = build_shared_mlp([3, 64, 128])
            self.local_fc = ResnetBlockFC(128 + 128 + 256, self.local_feature_dim)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return features.squeeze(-1)

    def forward_local(self, data, pts):
        assert self.local
        c, feature_list = self.forward_local_first_step(data)
        c, local_feats = self.forward_local_second_step(data, c, feature_list, pts)

        return c, local_feats

    def forward_local_first_step(self, data):
        assert self.local
        pointcloud = data[None]

        feature_list = []
        xyz, features = self._break_up_pc(pointcloud)
        feature_list.append((xyz, features))

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

            if xyz is not None:
                feature_list.append((xyz, features))

        c = features.squeeze(-1)
        return c, feature_list

    def forward_local_second_step(self, data, c, feature_list, pts):
        assert self.local
        B,_,_ = pts.size()

        if 'loc' in data:
            loc = data['loc'].view(B, 1, 3)
        else:
            loc = 0.

        if 'scale' in data:
            scale = 1.0 / data['scale'].view(B, 1, 1)
        else:
            scale = 1.

        radius = [0.1, 0.2, 0.4]
        n_sample = [16, 16, 8]
        local_feats = []
        for i, fl in enumerate(feature_list):
            xyz, features = fl[0], fl[1]

            xyz = (xyz - loc) * scale

            xyz_trans = xyz.transpose(2, 1).contiguous() # x: batch * 3 * n_x
            idx = ball_query(radius[i], n_sample[i], xyz, pts)

            if i == 0:
                grouped_xyz = grouping_operation(xyz_trans, idx)  # B * 3 * n_pts * n_sample
                grouped_xyz -= pts.transpose(1, 2).unsqueeze(-1)
                grouped_xyz = self.xyz_fc(grouped_xyz)

                local_feats.append(grouped_xyz.max(3)[0])
            else:
                grouped_features = grouping_operation(features, idx).max(3)[0] # B * C * n_pts
                local_feats.append(grouped_features) # B * C * n_pts

        local_feats = torch.cat(local_feats, 1)
        local_feats = self.local_fc(local_feats.transpose(1,2)) # B * n_pts * local_feat_dim

        return c, local_feats


class PointNet2MSGEncoder(PointNet2SSGEncoder):
    def _build_model(self):
        # different feature extractor
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[self.initial_feats_dim, 32, 32, 64], [self.initial_feats_dim, 64, 64, 128], [self.initial_feats_dim, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        )

        input_channels = 128 + 256 + 256
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        )

        if self.local:
            self.xyz_fc = build_shared_mlp([3, 64, 128])
            self.local_fc = ResnetBlockFC(128 + (64 + 128 + 128) + (128 + 256 + 256), self.local_feature_dim)
