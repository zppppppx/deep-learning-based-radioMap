import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from utils import Config



class weightGen_ob(nn.Module):
    """
    The nonlinear model in the Obstruction Network.
    """
    def __init__(self, opt=Config.Config()):
        super(weightGen_ob, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(opt.map_order, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, opt.map_order+1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = x.sum(-1).sum(-1)
        # print(x.shape)
        x = self.mlp(x)
        return x

class weightGen_rss(nn.Module):
    """
    The nonlinear model in the RM-Net.
    """
    def __init__(self, opt=Config.Config()):
        super(weightGen_rss, self).__init__()
        self.opt = opt
        self.mlp = nn.Sequential(
            nn.Linear(opt.map_order+1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32,32),
            nn.ReLU(inplace=True),
            nn.Linear(32, opt.map_order+1),
            nn.Softmax(dim=1)
        )

    def forward(self, rss_calc, rss_std):
        x = torch.abs(rss_calc - rss_std)
        return self.mlp(x)
