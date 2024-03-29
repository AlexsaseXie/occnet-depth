# TODO: implement MSN
# copy from MSN github, https://github.com/Colin97/MSN-Point-Cloud-Completion
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.point_completion.MSN_utils import expansion_penalty_module as expansion
from im2mesh.point_completion.MSN_utils import MDS_module
from im2mesh.point_completion.MSN_utils import emd_module as MSN_emd
from im2mesh.onet_depth.models.space_carver import SpaceCarverModule

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x

class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x

class MSN(nn.Module):
    def __init__(self, encoder, num_points = 8192, bottleneck_size = 1024, n_primitives = 16, 
                    space_carver_mode=False, space_carver_eps=3e-2):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives

        self.encoder = encoder
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0,self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()
        self.MSN_EMD = MSN_emd.emdModule()

        self.space_carver_mode = space_carver_mode
        if space_carver_mode:
            self.space_carver = SpaceCarverModule(mode=self.space_carver_mode, eps=space_carver_eps, training_drop_carving_p=1.)

    def forward(self, x, gt_pc=None, eps=None, it=None, reference=None, world_mat=None, camera_mat=None):
        # x : B * n_pts * 3
        x = x.transpose(2, 1)

        partial = x
        x = self.encoder(x)
        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.n_primitives)
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        outs = torch.cat(outs,2).contiguous() 
        out1 = outs.transpose(1, 2).contiguous() 
        
        dist, _, mean_mst_dis = self.expansion(out1, self.num_points//self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)

        id0 = torch.zeros(outs.shape[0], 1, outs.shape[2]).cuda().contiguous()
        outs = torch.cat( (outs, id0), 1)
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous()
        partial = torch.cat( (partial, id1), 1)
        xx = torch.cat( (outs, partial), 2)

        resampled_idx = MDS_module.minimum_density_sample(xx[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1], mean_mst_dis) 
        xx = MDS_module.gather_operation(xx, resampled_idx)
        delta = self.res(xx)
        xx = xx[:, 0:3, :] 
        out2 = (xx + delta).transpose(2,1).contiguous() 

        if gt_pc is not None:
            dist, _ = self.MSN_EMD(out1, gt_pc, eps, it)
            emd1 = dist.mean(1)

            dist, _ = self.MSN_EMD(out2, gt_pc, eps, it)
            emd2 = dist.mean(1)
            return emd1, emd2, loss_mst
        else:
            # eval phase space-carving
            if (not self.training) and self.space_carver_mode:
                assert reference is not None
                #_ = self.space_carver(out1, reference, cor_occ=None, world_mat=world_mat, camera_mat=camera_mat, replace=True)
                _ = self.space_carver(out2, reference, cor_occ=None, world_mat=world_mat, camera_mat=camera_mat, replace=True)
            return out1, out2, loss_mst