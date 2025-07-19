import torch
import torch.nn as nn
from .backbone import CSPBackbone
from .neck import FPNPAN
from .head import DetectionHead, ProtoNet

class YOLOSegModel(nn.Module):

    def __init__(self, model_cfg: dict, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_coeffs = model_cfg.get('num_coeffs', 32) 
        bb_channels = model_cfg.get('backbone_channels', [32, 64, 128, 256, 512])
        bb_depths = model_cfg.get('backbone_depths', [1, 2, 3, 1])
        neck_out = model_cfg.get('neck_out_channels', 256)
        proto_out_size = model_cfg.get('proto_out_size', [128, 128])

        # 1) Backbone
        self.backbone = CSPBackbone(channels=bb_channels, depths=bb_depths)

        # 2) Neck
        c3, c4, c5 = bb_channels[2], bb_channels[3], bb_channels[4]
        c6 = bb_channels[4]  # Downsample extra share same channels as c5
        self.neck = FPNPAN(channels=(c3, c4, c5, c6), out_channels=neck_out)

        self.head = DetectionHead(num_classes=self.num_classes, num_coeffs=self.num_coeffs,
                                  ch_in=(neck_out, neck_out, neck_out, neck_out))

        self.protonet = ProtoNet(c_in=neck_out, num_coeffs=self.num_coeffs,
                                 proto_channels=neck_out, out_size=proto_out_size)

    def forward(self, x):
        p3, p4, p5, p6 = self.backbone(x)
        n3, n4, n5, n6 = self.neck(p3, p4, p5, p6)
        head_out = self.head([n3, n4, n5, n6])
        proto = self.protonet(n3)

        preds = {f'P{i+3}': d for i, d in enumerate(head_out)}  # P3..P6

        return {
            'preds': preds,     
            'proto': proto   
        }

    def model_info(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'params_total': n_params, 'params_trainable': n_grad}