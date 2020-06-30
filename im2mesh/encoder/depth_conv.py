import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models

class Depth_Resnet18(nn.Module):
    r''' Depth_ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, use_linear=True, features_pretrained=True, model_pretrained=None):
        super().__init__()
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=features_pretrained)
        self.features.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.features.fc = nn.Sequential()

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

        if model_pretrained is not None:
            state_dict = torch.load(model_pretrained, map_location='cpu')
            self.load_state_dict(state_dict)

    def forward(self, x):
        #normalize
        net = x - 1.
        net = self.features(x)
        out = self.fc(net)
        return out