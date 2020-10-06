import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from im2mesh.layers import ResnetBlockFC
from im2mesh.common import normalize_imagenet, transform_points, project_to_camera


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(net))
        net = self.conv2(self.actvn(net))
        net = self.conv3(self.actvn(net))
        net = self.conv4(self.actvn(net))
        net = net.view(batch_size, 512, -1).mean(2)
        out = self.fc_out(self.actvn(net))

        return out


class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True, pretrained=True, local=False):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=pretrained)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

        self.local = local
        if self.local:
            self.local_fc = ResnetBlockFC(64+128+256+512, c_dim)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

    def forward_local(self, data, pts):
        assert self.local
        world_mat = data['world_mat']
        camera_mat = data['camera_mat']
        x = data[None]

        pts = transform_points(pts, world_mat)
        points_img = project_to_camera(pts, camera_mat)
        points_img = points_img.unsqueeze(1)

        local_feat_maps = []
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x) # 64 * 112 * 112
        
        x = self.features.layer1(x) 
        local_feat_maps.append(x) # 64 * 56 * 56
        x = self.features.layer2(x) 
        local_feat_maps.append(x) # 128 * 28 * 28
        x = self.features.layer3(x) 
        local_feat_maps.append(x) # 256 * 14 * 14
        x = self.features.layer4(x) 
        local_feat_maps.append(x) # 512 * 7 * 7

        x = self.features.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # get local feats
        local_feats = []
        for f in local_feat_maps:
            #f = f.detach()
            f = F.grid_sample(f, points_img, mode='nearest')
            f = f.squeeze(2)
            local_feats.append(f)

        local_feats = torch.cat(local_feats, dim=1)
        local_feats = local_feats.transpose(1, 2) # batch * n_pts * f_dim

        local_feats = self.local_fc(local_feats)

        # x: B * c_dim
        # local: feats B * n_pts * c_dim
        return x, local_feats


class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    forward = Resnet18.forward


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    forward = Resnet18.forward


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    forward = Resnet18.forward
