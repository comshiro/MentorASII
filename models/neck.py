import torch
import torch.nn as nn
from .layers import Conv, UpSample, Concat, C2f

class FPNPAN(nn.Module): # feature pyramid network + path aggregation network
    # top down - FPN, bottom up - PAN
    def __init__(self, channels=(128,256,512,512), out_channels=256):
        super().__init__()
        # channels tuple corespunde canalelor (P3, P4, P5, P6)
        c3, c4, c5, c6 = channels
        oc = out_channels

        self.reduce_p5 = Conv(c5, oc, k=1, s=1)
        self.reduce_p4 = Conv(c4, oc, k=1, s=1)
        self.reduce_p3 = Conv(c3, oc, k=1, s=1)
        self.reduce_p6 = Conv(c6, oc, k=1, s=1)

        self.upsample = UpSample(oc, oc)   # folosit pentru fiecare nivel sus
        self.concat = Concat(1)

        # C2f după concat (FPN sus)
        self.c2f_p5 = C2f(oc + oc, oc, n=1)  # (P5 up + P4)
        self.c2f_p4 = C2f(oc + oc, oc, n=1)
        self.c2f_p3 = C2f(oc + oc, oc, n=1)

        self.down_p3 = Conv(oc, oc, k=3, s=2)  # P3' -> spre P4''
        self.down_p4 = Conv(oc, oc, k=3, s=2)  # P4' -> spre P5''
        self.down_p5 = Conv(oc, oc, k=3, s=2)  # P5' -> spre P6''

        self.c2f_n4 = C2f(oc + oc, oc, n=1)
        self.c2f_n5 = C2f(oc + oc, oc, n=1)
        self.c2f_n6 = C2f(oc + oc, oc, n=1)

    def forward(self, p3, p4, p5, p6):
        p3r = self.reduce_p3(p3)
        p4r = self.reduce_p4(p4)
        p5r = self.reduce_p5(p5)
        p6r = self.reduce_p6(p6)


        # p6 -> up + p5
        u5 = self.upsample(p6r)
        p5u = self.c2f_p5(self.concat([u5, p5r]))  # P5'
        u4 = self.upsample(p5u)
        p4u = self.c2f_p4(self.concat([u4, p4r]))  # P4'
        u3 = self.upsample(p4u)
        p3u = self.c2f_p3(self.concat([u3, p3r]))  # P3'


        d4 = self.down_p3(p3u)            # down din P3'
        n4 = self.c2f_n4(self.concat([d4, p4u]))  # P4''
        d5 = self.down_p4(n4)
        n5 = self.c2f_n5(self.concat([d5, p5u]))  # P5''
        d6 = self.down_p5(n5)
        n6 = self.c2f_n6(self.concat([d6, p6r]))  # P6''

        # Returnez versiunile rafinate: P3', P4'', P5'', P6''
        return p3u, n4, n5, n6