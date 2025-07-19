import torch
import torch.nn as nn
from .layers import Conv

class DetectionHead(nn.Module):

    def init(self, num_classes=4, num_coeffs=32, ch_in=(256,256,256,256)):
        super().init()
        self.num_classes = num_classes # classes - features definitorii
        self.num_coeffs = num_coeffs

        self.stems = nn.ModuleList()
        self.box_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.coef_convs = nn.ModuleList() # weight

        for c in ch_in:
            # Stem mic: 1 conv pentru uniformizare
            self.stems.append(Conv(c, c, k=3, s=1))
            # Ramuri finale - conv 1x1 
            self.box_convs.append(nn.Conv2d(c, 4, kernel_size=1))
            self.cls_convs.append(nn.Conv2d(c, num_classes, kernel_size=1)) # x * w + b - calcul inainte de functia de activare
            self.coef_convs.append(nn.Conv2d(c, num_coeffs, kernel_size=1)) # masca de coeficienti

    def forward(self, feats):
        # feats: listă de tensori [P3', P4'', P5'', P6'']
        outputs = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            box = self.box_convs[i](x)   # (B, 4, H, W) - 4 - coordonate bbox
            cls = self.cls_convs[i](x)   # (B, C, H, W) - c - nr clase
            coef = self.coef_convs[i](x) # (B, P, H, W) - p - nr coeficienti
            outputs.append({"box": box, "cls": cls, "coef": coef})
        return outputs


class ProtoNet(nn.Module):

    def init(self, c_in=256, num_coeffs=32, proto_channels=256, out_size=(128,128)):
        super().init()
        self.out_h, self.out_w = out_size
        self.net = nn.Sequential(
            Conv(c_in, proto_channels, k=3, s=1),
            Conv(proto_channels, proto_channels, k=3, s=1),
            nn.Conv2d(proto_channels, num_coeffs, kernel_size=1)
        )

    def forward(self, x):
        p = self.net(x)  # (B, P, H', W') - batchsize - cate img procesezi simultan, channels - nr harti, h' height, w' width

        if p.shape[-2:] != (self.out_h, self.out_w):
            p = torch.nn.functional.interpolate(p, size=(self.out_h, self.out_w), mode='nearest')
        return p