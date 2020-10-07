import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from im2mesh import common
from im2mesh.layers import ResnetBlockFC

class Depth_Resnet18(nn.Module):
    r''' Depth_ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, use_linear=True, features_pretrained=True, model_pretrained=None, input_dim=1, normalize=False, local=False):
        super().__init__()
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=features_pretrained)
        self.normalize = normalize
        self.features.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

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

        if model_pretrained is not None:
            print('Loading depth encoder from ', model_pretrained)
            state_dict = torch.load(model_pretrained, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        if isinstance(x, dict):
            img = x['img']
            depth = x['depth']
            if self.normalize:
                depth = depth - 1.

            x = torch.cat((img, depth), dim = 1)
        # otherwise x is depth map already
        net = self.features(x)
        out = self.fc(net)
        return out

    def forward_local(self, data, pts):
        world_mat = data['world_mat']
        camera_mat = data['camera_mat']
        x = data[None]

        if isinstance(x, dict):
            img = x['img']
            depth = x['depth']
            if self.normalize:
                depth = depth - 1.

            x = torch.cat((img, depth), dim = 1)

        assert self.local
        pts = common.transform_points(pts, world_mat)
        points_img = common.project_to_camera(pts, camera_mat)
        points_img = points_img.unsqueeze(1)

        local_feat_maps = []
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
        

'''
resnet 18 forward:
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
'''
