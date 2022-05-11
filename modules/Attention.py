import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###########################################################################
# Not functional in the Main Frame  MAY BE USEFUL IN CONVOLUTIONAL RE-NET #
###########################################################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class spatial_attention(nn.Module):
    def __init__(self):
        super(spatial_attention, self).__init__()

        # conv
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, 1, 3),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x):
        """
        Args:
            x: here we need the Position map of User and UAV.
        """

        Position_avgpool = torch.mean(x, dim=1)[:,None,:]
        Position_maxpool = torch.max(x, dim=1)[0][:,None,:]
        
        Spatial_attention = self.conv(torch.cat((Position_maxpool, Position_avgpool), dim=1))
        Spatial_attention = torch.sigmoid(Spatial_attention)

        return Spatial_attention*x


class channel_attention(nn.Module):
    def __init__(self, in_ch, reduction_ratio=16, pool_types=['avg', 'max']):
        super(channel_attention, self).__init__()
        self.in_ch = in_ch
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_ch, in_ch//reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_ch//reduction_ratio, in_ch)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_raw = self.mlp(lp_pool)

            if channel_sum == None:
                channel_sum = channel_raw
            else:
                channel_sum += channel_raw

            scale = torch.sigmoid(channel_sum).unsqueeze(2).unsqueeze(3)
            return scale*x
        
class CBAM(nn.Module):
    def __init__(self, in_ch, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ch_att = channel_attention(in_ch, reduction_ratio, pool_types)
        self.spa_att = spatial_attention()

    def forward(self, x):
        x = self.ch_att(x)
        x = self.spa_att(x)

        return x


if __name__ == '__main__':
    x = torch.ones([1,64,94,94])
    ch_att = channel_attention(64)
    y = ch_att(x)
    sp_att = spatial_attention()
    y = sp_att(x)
    cbam = CBAM(64)
    y = cbam(x)
    print(y.shape)

 
