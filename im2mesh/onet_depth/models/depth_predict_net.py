import torch
from torch import nn
from im2mesh.onet_depth.models.hourglass import *
from im2mesh.onet_depth.models.uresnet import *

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class DepthPredictNet(nn.Module):
    ''' '''
    def __init__(self, n_hourglass=1, img_dim=3, inp_dim=128, oup_dim=1, bn=False, increase=0):
        super(DepthPredictNet, self).__init__()
        self.img_dim = img_dim
        self.inp_dim = inp_dim
        self.oup_dim = oup_dim
        self.n_hourglass = n_hourglass

        self.pre = nn.Sequential(
            #Conv(img_dim, inp_dim, 3, 1, bn=True, relu=True),
            #Conv(img_dim, 64, 7, 1, bn=True, relu=True),
            Conv(img_dim, 64, 3, 1, bn=True, relu=True),
            Residual(64, inp_dim)
        )

        self.hgs = nn.ModuleList(
            [ nn.Sequential(
                Hourglass(4, inp_dim, bn=bn, increase=increase)
            ) for i in range(n_hourglass) ]
        )

        self.features = nn.ModuleList( 
            [ nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(n_hourglass) ] 
        )

        self.outs = nn.ModuleList(
            [ Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(n_hourglass) ]
        )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(n_hourglass-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(n_hourglass-1)] )

    def forward(self, x):
        x = self.pre(x)
        combined_predicts = []
        for i in range(self.n_hourglass):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_predicts.append(preds)
            if i < self.n_hourglass - 1:
                x = x + self.merge_features[i](feature) + self.merge_preds[i](preds)
        return torch.stack(combined_predicts, 1)

    def get_last_predict(self, x):
        return self.forward(x)[:,self.n_hourglass-1]

class ViewAsLinear(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.shape[0], -1)

class UResnet_DepthPredict(nn.Module):
    def __init__(self, input_planes=3, out_planes=1, pred_min_max=True, num_layers=18):
        super(UResnet_DepthPredict, self).__init__()

        # Encoder
        module_list = list()
        assert num_layers in (18, 34, 50)
        if num_layers == 18:
            resnet = resnet18(pretrained=True)
            revresnet = revuresnet18(out_planes=out_planes)
        elif num_layers == 34:
            resnet = resnet34(pretrained=True)
            revresnet = revuresnet34(out_planes=out_planes)
        elif num_layers == 50:
            resnet = resnet50(pretrained=True)
            revresnet = revuresnet50(out_planes=out_planes)
        else:
            raise NotImplementedError
        
        in_conv = nn.Conv2d(input_planes, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        module_list.append(
            nn.Sequential(
                resnet.conv1 if input_planes == 3 else in_conv,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
        )
        module_list.append(resnet.layer1)
        module_list.append(resnet.layer2)
        module_list.append(resnet.layer3)
        module_list.append(resnet.layer4)
        self.encoder = nn.ModuleList(module_list)
        self.encoder_out = None

        # Decoder for depth image
        module_list = list()
        module_list.append(revresnet.layer1)
        module_list.append(revresnet.layer2)
        module_list.append(revresnet.layer3)
        module_list.append(revresnet.layer4)
        module_list.append(
            nn.Sequential(
                revresnet.deconv1,
                revresnet.bn1,
                revresnet.relu,
                revresnet.deconv2
            )
        )
        self.decoder = nn.ModuleList(module_list)

        # Decoder for depth_min, depth_max
        self.pred_min_max = pred_min_max
        if self.pred_min_max == True:
            module_list = nn.Sequential(
                nn.Conv2d(512, 512, 3, stride=3, padding=1),
                nn.Conv2d(512, 512, 3, stride=1),
                ViewAsLinear(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            )
            self.decoder_minmax = module_list
        
    
    def forward(self, x):
        # Encode
        feat = x
        feat_maps = list()
        for f in self.encoder:
            feat = f(feat)
            feat_maps.append(feat)
        self.encoder_out = feat_maps[-1]

        # Decode
        x = feat_maps[-1]
        for idx, f in enumerate(self.decoder):
            x = f(x)
            if idx < len(self.decoder) - 1:
                feat_map = feat_maps[-(idx + 2)]
                assert feat_map.shape[2:4] == x.shape[2:4]
                x = torch.cat((x, feat_map), dim=1)

        outputs = x.unsqueeze(1)

        if self.pred_min_max:
            self.predicted_min_max = self.decoder_minmax(self.encoder_out)
        return outputs

    def get_last_predict(self, x):
        return self.forward(x)[:,-1]

    def fetch_minmax(self):
        # must be called after forward or get_last_predict
        return self.predicted_min_max
