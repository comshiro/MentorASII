import torch
import torch.nn as nn
from .layers import Conv, C2f

class CSPBackbone(nn.Module):

    def init(self, channels=(32,64,128,256,512), depths=(1,2,3,1)):
        super().init()
        assert len(channels) == 5, "channels trebuie să aibă 5 valori (stem + 4 stage)"
        assert len(depths) == 4, "depths trebuie să aibă 4 valori (pt stage2..stage5)"

        c1, c2, c3, c4, c5 = channels
        d2, d3, d4, d5 = depths

        # Stem (downsample x2)
        self.stem = Conv(3, c1, k=3, s=2) #s - stride

        # Stage2: produce stride 4
        self.stage2 = nn.Sequential(
            Conv(c1, c2, k=3, s=2),  # downsample
            C2f(c2, c2, n=d2)
        )
        # Stage3: stride 8 (P3)
        self.stage3 = nn.Sequential(
            Conv(c2, c3, k=3, s=2),
            C2f(c3, c3, n=d3)
        )
        # Stage4: stride 16 (P4)
        self.stage4 = nn.Sequential(
            Conv(c3, c4, k=3, s=2),
            C2f(c4, c4, n=d4)
        )
        # Stage5: stride 32 (P5)
        self.stage5 = nn.Sequential(
            Conv(c4, c5, k=3, s=2),
            C2f(c5, c5, n=d5)
        )
        # Extra Downsample pentru stride 64 (P6) - doar un conv simplu
        self.down6 = Conv(c5, c5, k=3, s=2)

    def forward(self, x):
        x = self.stem(x)      # (B, c1, H/2, W/2)
        x2 = self.stage2(x)   # (B, c2, H/4, W/4)
        x3 = self.stage3(x2)  # (B, c3, H/8, W/8)   = P3
        x4 = self.stage4(x3)  # (B, c4, H/16, W/16) = P4
        x5 = self.stage5(x4)  # (B, c5, H/32, W/32) = P5
        x6 = self.down6(x5)   # (B, c5, H/64, W/64) = P6
        return x3, x4, x5, x6  # P3, P4, P5, P6