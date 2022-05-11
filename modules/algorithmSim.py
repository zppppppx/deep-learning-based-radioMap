import torch
import matplotlib.pyplot as plt
from modules.weightGen import *
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import Map, dist
from modules import Convolutional_RE_Net
from modules import Linear_RE_Net


# def barrier_check(Position_map, Obstacle_maps):
#     # Set inputs to the same shape to compare
#     pos = Position_map.repeat(1, Obstacle_maps.size(1), 1, 1)
#     obs = Obstacle_maps.repeat(Position_map.size(0), 1, 1, 1)


#     # Conventional techniques
#     # obs[pos == 0] = 0 # set the Null positions to zeros
#     # obs[pos >= obs] = 0 # set the positions where there is no obstruct to zeros
#     # obs[obs != 0] = 1 # set the positions where there is obstruct to ones

#     diff = obs*pos - pos*pos
#     out = F.relu(diff)

#     return out


def barrier_check_pos(Position_map, Obstacle_maps):
    """
    This function realizes the function to generate the obstruction indications (see the obstruction network).
    pos means positive, which is to say we obtain the obstructed elements from those of the Obstacle Maps higher than 
    those of the Position Map.

    #################################
    # We mainly adopt this solution #
    #################################

    Args:
        Position_map: generated from the location pairs.
        Obstacle_maps: generated from the RE-Net.

    Return:
        out: obstruction relationships turned into a vector, see the Obstruction Net part.
    """
    # Set inputs to the same shape to compare
    pos = Position_map.repeat(1, Obstacle_maps.size(1), 1, 1)
    obs = Obstacle_maps.repeat(Position_map.size(0), 1, 1, 1)

    # Conventional techniques
    # obs[pos == 0] = 0 # set the Null positions to zeros
    # obs[pos >= obs] = 0 # set the positions where there is no obstruct to zeros
    # obs[obs != 0] = 1 # set the positions where there is obstruct to ones

    diff = obs*pos - pos*pos
    out = F.relu(diff)

    out = out.sum(-1).sum(-1)

    return out

def barrier_check_neg(Position_map, Obstacle_maps):
    """
    This function realizes the function to generate the obstruction indications (see the obstruction network).
    neg means negative, which is to say we obtain the obstructed elements from those of the Position Map lower than 
    those of the Obstacle Map.

    Args:
        Position_map: generated from the location pairs.
        Obstacle_maps: generated from the RE-Net.

    Return:
        out: obstruction relationships turned into a vector, see the Obstruction Net part.
    """
    pos = Position_map.repeat(1, Obstacle_maps.size(1), 1, 1) 
    obs = Obstacle_maps.repeat(Position_map.size(0), 1, 1, 1)

    diff = F.relu(pos-obs)*pos + obs*pos
    diff = torch.abs(diff - pos*pos)
    out = diff.sum(-1).sum(-1)

    return out


######################
# Proposed Framework #
######################

class Algorithm(nn.Module):
    """
    Args:
        x: location pair.
        map: metainfo, stacked fundamental shapes for Convolutional RE-Net and 1 for Linear RE-Net.

    Return:
        out: predicted RSS.
        obs_weight: tell the propagation condition, namely Sk, generated the Obstruction Network.
        Obstacle_maps: the virtual obstacle map.
        multi-channel gain: 
    """
    def __init__(self, opt, scale=50):
        super(Algorithm, self).__init__()
        self.opt = opt
        self.scale = scale
        self.params_model = nn.Linear(2, opt.map_order+1, bias=False) # Multichannel Gain.
        self.map = Convolutional_RE_Net.map_generate(opt) if opt.RE_Net == 'Conv' else Linear_RE_Net.map_generate(opt) # RE-Net
        self.weight_gen_ob = weightGen_ob(opt) # nonlinear model in the Obstruction Network.


        ##################################################
        # Initialize the parameters in Multichannel Gain #
        ##################################################

        # self.params_model.weight = torch.nn.Parameter(torch.tensor([[-16.,-35.],[-24.,-27.],[-36.,-18.],[-55., 12.]]), requires_grad=True)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor([[-22.,-27.],[-28.,-23.],[-36.,-21.]]), requires_grad=True)

        self.params_model.weight = torch.nn.Parameter(torch.tensor([[-22.,-27.],[-28.,-24.],[-36.,-23.]]), requires_grad=True)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor([[-22.,-28.],[-36.,-22.]]), requires_grad=True)


        # self.params_model.weight = torch.nn.Parameter(torch.tensor([[-22.,-27.],[-28.8,-22.],[-36.,-21.8]]), requires_grad=True)


        # G/P(N)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor([[-21.,-30.],[-38.,-20.]]), requires_grad=True)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor([[-23.28,-23.259],[-33.492,-24.262],[-43.577,-12.865]]), requires_grad=True)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor(
        #     [[-22.,-28.],[-27.,-26.],[-32.,-24.],[-36., -22.]]), requires_grad=True)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor(
            # [[-28., -24.],[-28., -24.],[-28., -24.],[-28., -24.],[-28., -24.]]), requires_grad=True) # 5
        # self.params_model.weight = torch.nn.Parameter(torch.tensor(
        #     [[-33.,-16.3],[-33.,-17.3],[-33.,-18.3],[-33.,-19.3],[-33.,-20.3]]), requires_grad=True)
        # self.params_model.weight = torch.nn.Parameter(torch.tensor(
        #     [[-33.,-10.3],[-33.,-12.3],[-33.,-14.3],[-33.,-16.3],[-33.,-18.3],
        #     [-33.,-20.3],[-33.,-22.3],[-33.,-24.3],[-33.,-26.3],[-33.,-28.3]]), requires_grad=True)


    def forward(self, x, mapseed):
        device = x.device

        # Branch 1: The distance param models.
        log_dist = dist.distance(x.cpu()).to(device)
        param_output = self.params_model(log_dist)

        # Branch 2: Generate the map and the final barrier map
        Position_maps = Map.locs_to_map(x.cpu(), self.opt)/self.scale
        Position_maps = Position_maps.to(device)
        Obstacle_maps = self.map(mapseed)
        
        Barrier_vec = barrier_check_pos(Position_maps, Obstacle_maps) # 
        obs_weight = self.weight_gen_ob(Barrier_vec) # The obstruction degree

        # calculate the model's output
        out = param_output*obs_weight
        out = out.sum(-1)

        return out, obs_weight, Obstacle_maps, param_output


