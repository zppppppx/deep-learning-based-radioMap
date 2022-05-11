
from modules.weightGen import *
from modules.algorithmSim import barrier_check_pos
import utils.Config as Config
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import numpy as np


def std_out_ob(barrier_vec, opt):
    """
    This is standard obstacle map output, designed to train the nonlinear model in the Obstruction Network.

    Args:
        barrier_vec: the obstruction indications turned from a tensor to a vector (see the Obstruction Net part)
        opt: Config.Config(). Configures.

    Return:
        out_std: artificial labels.
    """
    batch_size = barrier_vec.size(0)

    out_std = torch.zeros([batch_size, opt.map_order+1])
    for i in range(batch_size):
        for j in range(opt.map_order):
            check_line = F.relu(barrier_vec[i, opt.map_order-j-1]-0)
            if check_line > 0:
                out_std[i, opt.map_order-j] += 1
                break

        if out_std[i].sum(-1) == 0:
            out_std[i][0] = 1

    return out_std

def std_out_rss(rss_calc, rss_std, opt):
    """
    This is standard rss output, designed to provide labels to train the nonlinear model in the RM-Net.

    Args:
        rss_calc: calculated rss.
        rss_std: standard rss.

    Return:
        out_std: standard category.
    """
    diff = torch.abs(rss_calc - rss_std)
    args = torch.argmin(diff, dim=1)
    out_std = torch.zeros(rss_std.size(0), opt.map_order+1)
    for i in range(rss_std.size(0)):
        out_std[i, args[i]] += 1

    return out_std



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = Config.Config()
scale = 1

feature_net = weightGen_ob(opt).to(device)
net_path_ob = './weight_ob_' + str(opt.map_order) + '.pth'


if os.path.exists(net_path_ob):
    state = torch.load(net_path_ob)
    feature_net.load_state_dict(state)


optimizer = optim.Adam([p for p in feature_net.parameters()], lr=1e-2)
criterion = nn.MSELoss().to(device)

# Training for obstacle
running_loss = 0.
for i in range(30):
    barriers_vec = torch.tensor([])

    for j in range(opt.map_order):
        bar = torch.rand([opt.batchsize, opt.map_order])
        bar[:, j:] = 0
        barriers_vec = torch.cat((barriers_vec, bar), dim=0)
        # print(i, '\t', bar)
    bar = torch.rand([opt.batchsize, opt.map_order])
    barriers_vec = torch.cat((barriers_vec, bar), dim=0)
    # print(barriers_vec)
    rand_index = np.random.randint(0,opt.batchsize*(opt.map_order+1), opt.batchsize*(opt.map_order+1))
    # print(rand_index)
    barriers_vec = barriers_vec[rand_index]
    barriers_vec = barriers_vec.to(device)


    # print(barriers.shape)

    # plt.subplot(2,2,1)
    # plt.imshow(Position_map[0,0])
    # plt.subplot(2,2,2)
    # plt.imshow(Obstacle_map[0,0])
    # plt.subplot(2,2,3)
    # plt.imshow(barriers[0,0].cpu().detach())
    # plt.show()

    out_std = std_out_ob(barriers_vec, opt).to(device)
    # print(out_std)

    optimizer.zero_grad()
    out = feature_net(barriers_vec)
    loss = criterion(out, out_std)
    loss.backward()

    optimizer.step()
    running_loss += loss.item()

    print('epoch: %d, loss: %.4f' % (i, running_loss))
    print(out[0])

    running_loss = 0
    
    state = feature_net.state_dict()
    torch.save(state, net_path_ob)

state = feature_net.state_dict()
torch.save(state, net_path_ob)

# Training for rss
feature_net = weightGen_rss(opt).to(device)
net_path_rss = './weight_rss_' + str(opt.map_order) + '.pth'

if os.path.exists(net_path_rss):
    state = torch.load(net_path_rss)
    feature_net.load_state_dict(state)


optimizer = optim.Adam([p for p in feature_net.parameters()], lr=1e-2)
criterion = nn.MSELoss().to(device)


running_loss = 0.
for i in range(200):
    rss_calc = -100*torch.rand(opt.batchsize, opt.map_order+1).to(device)
    rss_std = -100*torch.rand(opt.batchsize, 1).to(device)
    out_std = std_out_rss(rss_std, rss_calc, opt).to(device)
    # print(out_std)

    optimizer.zero_grad()
    out = feature_net(rss_calc, rss_std)
    loss = criterion(out, out_std)
    loss.backward()

    optimizer.step()
    running_loss += loss.item()

    # print('epoch: %d, loss: %.4f' % (i, running_loss))
    # print(out[0])

    running_loss = 0
    
    state = feature_net.state_dict()
    torch.save(state, net_path_rss)

state = feature_net.state_dict()
torch.save(state, net_path_rss)