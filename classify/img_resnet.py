from im2mesh import config, data, common
from torch import nn
from torchvision import transforms
from torchvision import models
from im2mesh.encoder.conv import Resnet18

class ImgClassify_ResNet18(nn.Module):
    def __init__(self, num_classes=13, c_dim=512):
        super(ImgClassify_ResNet18, self).__init__()
        self.normalize = True
        self.features = Resnet18(c_dim, normalize=True, use_linear=True)
        self.fc = nn.Linear(c_dim, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, data, device):
        inputs = data.get('inputs').to(device)
        out = self.features(inputs)
        out = self.fc(out)
        return out 
    
    def get_loss(self, data, device):
        class_gt = data.get('category').to(device)
        out = self.forward(data, device) 
        loss = self.loss_func(out, class_gt)
        return loss

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model