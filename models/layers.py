from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    if p is None:
        if isinstance(k, int):
            return k // 2
        else:  # list/tuple
            return [x // 2 for x in k]
    return p


class Conv(nn.Module):

    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module): # ajuta sa propagam gradientul si sa crestem adancimea retelei
    def __init__(self, c_in, c_out, shortcut=True, expansion=0.5):
        super().__init__()
        hidden = int(c_out * expansion) # nr canale interne / bottleneck width
        self.cv1 = Conv(c_in, hidden, k=1, s=1) #comrpimam blocul -> 
        self.cv2 = Conv(hidden, c_out, k=3, s=1) #procesam -> restauram
        self.use_add = shortcut and c_in == c_out

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.use_add else y


class C2f(nn.Module): # C2f = Cross Stage Partial Fused

    def __init__(self, c_in, c_out, n=1, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c_in, c_out, k=1, s=1)
        half = c_out // 2
        # n Bottleneck-uri care procesează DOAR jumătate din canale
        self.blocks = nn.Sequential(*[Bottleneck(half, half, shortcut, expansion=1.0) for _ in range(n)])
        self.cv2 = Conv(c_out, c_out, k=1, s=1)

    def forward(self, x):
        y = self.cv1(x)
        y1, y2 = y.chunk(2, dim=1)
        y1 = self.blocks(y1)
        out = torch.cat([y1, y2], dim=1)
        return self.cv2(out)


class UpSample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = Conv(c_in, c_out, k=1, s=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.conv(x)


class Concat(nn.Module): #concateneaza harti de features de aceeasi dimensiune pe axa canalelor(dim din tensor care stocheaza hartile)
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs: List[torch.Tensor]):
        return torch.cat(xs, dim=self.dim)
