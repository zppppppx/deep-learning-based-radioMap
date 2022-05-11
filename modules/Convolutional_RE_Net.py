from utils.Config import Config
import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


#########################################
# Basic blocks mentioned in the Appendix #
#########################################
class double_conv(nn.Module):
    """
    Double-Conv Layer.
    """
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(self.init_weights)
    
    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal_(m.weight)
            # init.uniform_(m.weight)
            init.constant_(m.bias,0)

class up_net_sole(nn.Module):
    """
    Up-sampling Deconvolutional Layer.
    """
    def __init__(self, in_ch, out_ch):
        super(up_net_sole, self).__init__()
        # self.up = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(in_ch, in_ch, 3, 1, 1),
        #     nn.ReLU()
        # )
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, 2)
        self.conv = double_conv(in_ch, out_ch)
        self.up.apply(self.init_weights)
        self.conv.apply(self.init_weights)

    def forward(self, x1, target_size):
        x1 = self.up(x1)

        diffY = target_size[0] - x1.size()[2]
        diffX = target_size[1] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2), 'replicate')

        # x = torch.cat([x2,x1], dim=1)
        x = self.conv(x1)
        return x
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal_(m.weight)
            init.constant_(m.bias,0)



#############################################
# Overall structure of Convolutional RE-Net #
#############################################
class map_block(nn.Module):
    """
    Single structure mentioned in the Convolutional RE-Net.
    """
    def __init__(self, opt):
        super(map_block, self).__init__()
        self.map_size = opt.map_size
        self.map_order = opt.map_order
        self.target_size = [
            [*map(lambda x: x//2//2, self.map_size)],
            [*map(lambda x: x//2, self.map_size)],
            [*map(lambda x: x, self.map_size)],
        ]

        self.inconv = nn.Sequential(
            double_conv(opt.shapeUnits.num, 128),
            double_conv(128, 128),
            double_conv(128, 256),
            )
        # self.inconv = nn.Sequential(
        #     double_conv(opt.shapeUnits.num, 64),
        #     double_conv(64, 64),
        #     double_conv(64, 128),
        #     )
        

        self.up1 = up_net_sole(256, 256) # up_net 1
        self.up2 = up_net_sole(256, 256) # up_net 2
        self.up3 = up_net_sole(256, 512) # up_net 3

        # self.up1 = up_net_sole(128, 256) # up_net 1
        # self.up2 = up_net_sole(256, 256) # up_net 2
        # self.up3 = up_net_sole(256, 512) # up_net 3

        # self.outconv = nn.Sequential(
        #     double_conv(512, 256),
        #     double_conv(256, 128),
        #     double_conv(128, 64)
        # )
        self.outconv = nn.Sequential(
            double_conv(512, 256),
            double_conv(256, 128),
            double_conv(128, 64),
        )
        self.outcome = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # Shallow Convolutional Network
        x = self.inconv(x)

        # Up-sampling Deconvolutional Network
        x = self.up1(x, self.target_size[0])
        x = self.up2(x, self.target_size[1])
        x = self.up3(x, self.target_size[2])

        # Shallow Convolutional Averaging Network
        x = self.outconv(x)

        # Convolutional Layer
        x = self.outcome(x)

        return x


class map_generate(nn.Module):
    """
    Combine the K identical structures.
    """
    def __init__(self, opt):
        super(map_generate, self).__init__()
        self.map_blocks = nn.ModuleList([map_block(opt) for _ in range(opt.map_order)])
        self.opt = opt

    def forward(self, x):
        device = x.device
        out = torch.zeros(1, self.opt.map_order, self.opt.map_size[0], self.opt.map_size[1]).to(device)
        for i in range(self.opt.map_order):
            out[0, i] = self.map_blocks[i](x)[0,0]

        return out


