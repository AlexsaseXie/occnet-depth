from torch import nn
from im2mesh.onet_depth.models import background_setting
from im2mesh.onet_depth.training import compose_inputs
from im2mesh.encoder.pointnet import feature_transform_reguliarzer
from im2mesh.encoder import encoder_dict

class BaseClassifyModel(nn.Module):
    def __init__(self, encoder=None, num_classes=13, c_dim=512):
        super(BaseClassifyModel, self).__init__()

        self.features = encoder
        self.pred_fc = nn.Linear(c_dim, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def get_inputs(self, data, device):
        raise NotImplementedError

    def forward(self, data, device):
        encoder_inputs = self.get_inputs(data, device)

        out = self.features(encoder_inputs)
        if isinstance(out, tuple):
            out = out[0]

        out = self.pred_fc(out)
        return out
    
    def get_loss(self, data, device):
        class_gt = data.get('category').to(device)

        out = self.forward(data, device)
        loss = self.loss_func(out, class_gt)
        return loss

class ImgClassifyModel(BaseClassifyModel):
    def get_inputs(self, data, device):
        inputs = data.get('inputs').to(device)
        return inputs

class DepthClassifyModel(BaseClassifyModel):
    def __init__(self, encoder=None, num_classes=13, c_dim=512, with_img=False):
        super(DepthClassifyModel, self).__init__(encoder=encoder, num_classes=num_classes, c_dim=c_dim)
        self.with_img = with_img 

    def get_inputs(self, data, device):
        gt_depth_maps = data.get('inputs.depth').to(device)
        gt_mask = data.get('inputs.mask').to(device).byte()
        background_setting(gt_depth_maps, gt_mask)
        encoder_inputs = gt_depth_maps

        if self.with_img:
            img = data.get('inputs').to(device)
            encoder_inputs = {'img':img, 'depth': encoder_inputs}

        return encoder_inputs

class DepthPointcloudClassifyModel(BaseClassifyModel):
    def __init__(self, encoder=None, num_classes=13, c_dim=512, depth_pointcloud_transfer='world'):
        super(DepthPointcloudClassifyModel, self).__init__(encoder=encoder, num_classes=num_classes, c_dim=c_dim)
        self.depth_pointcloud_transfer = depth_pointcloud_transfer 

    def get_inputs(self, data, device):
        pc, _ = compose_inputs(data, mode='train', device=device, input_type='depth_pointcloud', depth_pointcloud_transfer=self.depth_pointcloud_transfer)
        return pc

class PointcloudClassify_Pointnet(DepthPointcloudClassifyModel):
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

def get_model(self, input_type, cfg):
    encoder_type = cfg['model']['encoder']
    c_dim = cfg['model']['c_dim']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    encoder = encoder_dict[encoder_type](
            c_dim=c_dim,
            **encoder_kwargs
        )

    if input_type == 'img':
        model = ImgClassifyModel(encoder=encoder, num_classes=13, c_dim=c_dim)
    elif input_type == 'img_with_depth':
        if 'pred_with_img' in cfg['model']:
            pred_with_img = cfg['model']['pred_with_img']
        else:
            pred_with_img = False

        model = DepthClassifyModel(encoder=encoder, num_classes=13, c_dim=c_dim, with_img=pred_with_img)
    elif input_type == 'depth_pointcloud':
        if 'depth_pointcloud_transfer' in cfg['model']:
            depth_pointcloud_transfer = cfg['model']['depth_pointcloud_transfer']
        else:
            depth_pointcloud_transfer = 'world'

        if encoder_type == 'pointnet':
            # special case for pointnet
            model = PointcloudClassify_Pointnet(encoder=encoder, num_classes=13, c_dim=c_dim, depth_pointcloud_transfer=depth_pointcloud_transfer)
        else:
            model = DepthPointcloudClassifyModel(encoder=encoder, num_classes=13, c_dim=c_dim, depth_pointcloud_transfer=depth_pointcloud_transfer)
    else:
        raise NotImplementedError

    return model
    