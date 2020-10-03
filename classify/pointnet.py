from im2mesh import common
from torch import nn
from im2mesh.encoder.pointnet import PointNetEncoder, feature_transform_reguliarzer
from im2mesh.onet_depth.training import compose_inputs

class PointcloudClassify_Pointnet(nn.Module):
    def __init__(self, num_classes=13, c_dim=512, depth_pointcloud_transfer=None):
        super(PointcloudClassify_Pointnet, self).__init__()
        self.features = PointNetEncoder(c_dim=c_dim, global_feat=True, feature_transform=True, channel=3, only_point_feature=False)
        self.pred_fc = nn.Linear(c_dim, num_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self.depth_pointcloud_transfer = depth_pointcloud_transfer

    def get_inputs(self, data, device):
        pc, _ = compose_inputs(data, mode='train', device=device, input_type='depth_pointcloud', depth_pointcloud_transfer=self.depth_pointcloud_transfer)
        return pc

    def forward(self, data, device):
        pc = self.get_inputs(data, device)

        out, trans_point, trans_feat = self.features(pc)
        out = self.pred_fc(out)
        return out, trans_point, trans_feat

    def get_loss(self, data, device):
        class_gt = data.get('category').to(device)

        out, _, trans_feat = self.forward(data, device)
        loss = self.loss_func(out, class_gt)
        loss = loss + 0.001 * feature_transform_reguliarzer(trans_feat)
        return loss

    
    def to(self, device):
        model = super().to(device)
        model._device = device
        return model
