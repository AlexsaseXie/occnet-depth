from im2mesh import config, data, common
from torch import nn
from torchvision import transforms
from torchvision import models

from im2mesh.encoder.depth_conv import Depth_Resnet18
from im2mesh.onet_depth.models import background_setting

class DepthClassify_Resnet18(nn.Module):
    def __init__(self, num_classes=13, c_dim=512, pretrained=True, with_img=False):
        super(DepthClassify_Resnet18, self).__init__()
        self.with_img = with_img
        input_dim = 1
        if with_img:
            input_dim = 4
        self.features = Depth_Resnet18(c_dim = c_dim, use_linear=True, features_pretrained=pretrained, input_dim=input_dim)
        self.pred_fc = nn.Linear(c_dim, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, data, device):
        gt_depth_maps = data.get('inputs.depth').to(device)
        gt_mask = data.get('inputs.mask').to(device).byte()
        background_setting(gt_depth_maps, gt_mask)
        encoder_inputs = gt_depth_maps

        if self.with_img:
            img = data.get('inputs').to(device)
            encoder_inputs = {'img':img, 'depth': encoder_inputs}

        out = self.features(encoder_inputs)
        out = self.pred_fc(out)
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
