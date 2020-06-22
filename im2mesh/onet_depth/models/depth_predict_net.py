import torch
from torch import nn
from im2mesh.onet_depth.models.hourglass import *

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class DepthPredictNet(nn.Module):
    ''' '''
    def __init__(self, n_hourglass=1, img_dim=3, inp_dim=128, oup_dim=1, bn=False, increase=0):
        self.img_dim = img_dim
        self.inp_dim = inp_dim
        self.oup_dim = oup_dim
        self.n_hourglass = n_hourglass

        self.pre = nn.Sequential(
            Conv(img_dim, 64, 3, 1, bn=True, relu=True),
            Residual(64, inp_dim),
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


