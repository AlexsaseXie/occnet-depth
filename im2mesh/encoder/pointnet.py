import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC, ResnetBlockConv1d
import numpy as np
import torch.nn.functional as F

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

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
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
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

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


## copied from pytorch-pointnet

class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(
            np.float32)).view(1, 9).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(
            np.float32)).view(1, self.k * self.k).repeat(batchsize, 1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, c_dim=1024, global_feat=True, feature_transform=True, channel=3, only_point_feature=False, model_pretrained=None, local=False):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, c_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(c_dim)

        self.c_dim = c_dim
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.only_point_feature = only_point_feature
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        if model_pretrained is not None:
            print('Loading depth encoder from ', model_pretrained)
            state_dict = torch.load(model_pretrained, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

        self.local = local
        if self.local:
            self.local_fc = ResnetBlockFC(64+128+c_dim, c_dim)

    def forward(self, x):
        # match the input of pointnet
        x = x.transpose(2, 1) # x: batch * 3 * n_pts
        B, D, N = x.size()

        trans_point = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans_point)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.c_dim)

        if not self.global_feat:
            x = x.view(-1, self.c_dim, 1).repeat(1, 1, N)
            x = torch.cat([x, pointfeat], 1)

        if self.only_point_feature:
            return x
        else:
            return x, trans_point, trans_feat

    def forward_local(self, data, pts, K=5):
        #TODO: this function needs rework
        assert self.local
        x = data[None]

        # calc dist matrix
        B, N, D = x.size()
        _, N_pt, _ = pts.size()
        
        if D > 3:
            a = x[:,:,:3].unsqueeze(2).expand(B, N, N_pt, 3)
            b = pts[:,:,:3].unsqueeze(1).expand(B, N, N_pt, 3)
        else:
            a = x.unsqueeze(2).expand(B, N, N_pt, D)
            b = pts.unsqueeze(1).expand(B, N, N_pt, D)
        dist = ((a - b) ** 2).sum(dim=3)
        
        _, indices = torch.topk(dist, K, dim=1, largest=False) # indices : B * K * N_pt
        # e.g. dist_v == dist.gather(1, indices)

        indices = indices.unsqueeze(3).transpose(1, 3) # indices : B * 1 * N_pt * K

        feature_maps = []
        # match the input of pointnet
        x = x.transpose(2, 1) # x: batch * 3 * n_pts

        trans_point = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans_point)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        feature_maps.append(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        feature_maps.append(x)
        x = self.bn3(self.conv3(x))
        feature_maps.append(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.c_dim)

        if not self.global_feat:
            x = x.view(-1, self.c_dim, 1).repeat(1, 1, N)
            x = torch.cat([x, pointfeat], 1)

        local_feats = []
        for fm in feature_maps:
            #fm = fm.detach()
            _, fm_c, _ = fm.size() # B * fm_c * N
            
            fm = fm.unsqueeze(2).expand(B, fm_c, N_pt, N)
            f_local = fm.gather(3, indices.expand(B, fm_c, N_pt, K))
            local_feats.append(f_local.mean(dim=3)) # append: B * fm_c * N_pt

        local_feats = torch.cat(local_feats, 1).transpose(2, 1)
        local_feats = self.local_fc(local_feats)

        if self.only_point_feature:
            return x, local_feats
        else:
            return x, local_feats, trans_point, trans_feat


class PointNetResEncoder(nn.Module):
    def __init__(self, c_dim=1024, global_feat=True, feature_transform=True, channel=3, only_point_feature=False, model_pretrained=None):
        super(PointNetResEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.block1 = ResnetBlockConv1d(channel, 64, 64)
        self.block2 = ResnetBlockConv1d(64, 128, 128)
        self.block3 = ResnetBlockConv1d(128, c_dim, c_dim)

        self.c_dim = c_dim
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.only_point_feature = only_point_feature
        if self.feature_transform:
            self.fstn = STNkd(k=64)

        if model_pretrained is not None:
            print('Loading depth encoder from ', model_pretrained)
            state_dict = torch.load(model_pretrained, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, x):
        # match the input of pointnet
        x = x.transpose(2, 1)

        B, D, N = x.size()
        trans_point = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans_point)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = self.block1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.block2(x)
        x = self.block3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.c_dim)

        if not self.global_feat:
            x = x.view(-1, self.c_dim, 1).repeat(1, 1, N)
            x = torch.cat([x, pointfeat], 1)

        if self.only_point_feature:
            return x
        else:
            return x, trans_point, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]

    if trans.is_cuda:
        I = I.cuda()
    # mse loss
    loss = torch.pow(torch.bmm(trans, trans.transpose(2, 1)) - I, 2).mean()
    return loss


class StackedPointnet(nn.Module):
    def __init__(self, c_dim=1024, channel=3):
        self.c_dim = c_dim
        self.channel = channel

        self.stn = STN3d(channel)

        self.pn1 = nn.ModuleList(
            nn.Conv1d(channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )

        self.pn2 = nn.ModuleList([
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
        ])

    def forward(self, x):
        # match the input of pointnet
        x = x.transpose(2, 1) # x: batch * 3 * n_pts
        B, D, N = x.size()

        # STN3d
        trans_point = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans_point)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)

        feat = self.pn1(x)
        feat_global = torch.max(feat, 2, keepdim=True)[0]
        feat_global = feat_global.repeat((1,1,N))
        feat = torch.cat(([feat, feat_global]), dim=1)

        feat = self.pn2(x)
        feat_global = torch.max(feat, 2, keepdim=True)[0]
        feat_global = feat_global.view(-1, self.c_dim)

        return feat_global


class MSNPointNetFeat(nn.Module):
    def __init__(self, num_points = 8192, global_feat = True, bottleneck_size=1024):
        super(MSNPointNetFeat, self).__init__()
        self.stn = STN3d(channel=3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

        self.bottleneck_size = bottleneck_size
        self.li = nn.Linear(1024, self.bottleneck_size),
        self.li_bn = nn.BatchNorm1d(self.bottleneck_size),
        self.li_relu = nn.ReLU()
    def forward(self, x):
        # x: B * 3 * n_pts
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = self.li_relu(self.li_bn(self.li(x)))
        return x