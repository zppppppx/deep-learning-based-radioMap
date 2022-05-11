import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
      

class linear_unit(nn.Module):
    """
    The basic linear layer.
    """
    def __init__(self, ch):
        super(linear_unit, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, ch, bias=False),
            # nn.Linear(1, 16),
            # nn.Linear(16,ch),
            nn.ReLU(inplace=True)
        )
        self.fc.apply(self.init_weight)
        
    def forward(self, x):
        return self.fc(x)

    @staticmethod
    def init_weight(m):
        if type(m) == nn.Linear:
            # init.uniform_(m.weight, 0, 0.1)
            init.constant_(m.weight, 0.1) # initialize


class map_generate(nn.Module):
    """
    Reshape the linear layers to form a obstacle map.
    """
    def __init__(self, opt):
        super(map_generate, self).__init__()
        self.opt = opt
        self.units = self._set_units(opt.map_size[0]*opt.map_size[1], opt.map_order)
        # self.map_seed = torch.ones(1, 1, opt.map_size[0], opt.map_size[1])

    def forward(self, x):
        map_seed = x.to(self.opt.device)

        return self._units_calc(self.units, map_seed, self.opt)

    def _set_units(self, num, map_order, unit=linear_unit):
        """
        Set corresponding number of the linear layers.

        Args:
            num: the number of the linear layers
        """
        units = nn.ModuleList([unit(map_order) for i in range(num)])
        return units
    
    def _units_calc(self, units, x, opt):
        """
        Obtain each linear layer's output.
        """
        out = torch.zeros(opt.map_size[0]*opt.map_size[1], opt.map_order).to(x.device)
        # print(x.shape)
        for i in range(len(units)):
            out_i = units[i](x)
            # print(out_i.shape)
            # out = torch.cat((out, out_i), dim=2)
            out[i] = out_i
        
        return out.permute([1,0])[None,].view(1, opt.map_order, opt.map_size[0], opt.map_size[1])

    @staticmethod
    def units_loss(out, std, loss_func, opt):
        loss = torch.zeros([1]).to(opt.device)
        for i in range(len(out)):
            lossi = loss_func(out[i].to(opt.device), std[i].to(opt.device))
            loss += lossi
        return loss

