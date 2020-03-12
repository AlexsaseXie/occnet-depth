import torch.nn as nn
import torch
# import torch.nn.functional as F
from im2mesh.onet_multi_layers_predict.models import resnet
from im2mesh.common import normalize_imagenet

class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = resnet.resnet18(pretrained=True)

        if c_dim != 512:
            self.fc3 = nn.Linear(512, c_dim)
            self.fc2 = nn.Linear(512, c_dim)
            self.fc1 = nn.Linear(512, c_dim)
        else:
            self.fc3 = nn.Sequential()
            self.fc2 = nn.Sequential()
            self.fc1 = nn.Sequential()

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        f3,f2,f1 = self.features(x)
        f3 = self.fc3(f3)
        f2 = self.fc2(f2)
        f1 = self.fc1(f1)
        #f = torch.cat([f3,f2,f1], dim=1)
        return f3, f2, f1