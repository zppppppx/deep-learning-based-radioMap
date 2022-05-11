import os
import numpy as np
import torch
from dataset import locPair
from modules import algorithmSim, weightGen
import utils.Config as Config
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.load_dict import load_dict



opt = Config.Config()
file_path = '../data/radiomap_simulated100tx_3class_noise.mat'
# file_path = '../data/radiomap_simulated100tx.mat'
net_path = opt.RE_Net + '_3_' + str(opt.map_order+1) + '.pth' # 3 for 3class. Suit yourself.

radio_net = algorithmSim.Algorithm(opt).to(opt.device)
radio_data_val = locPair.RadioMap(file_path, fraction=1, seed=5)


if os.path.exists(net_path):
    state = torch.load(net_path)
    radio_net.load_state_dict(state['net'])
print(radio_net.params_model.state_dict())

criterion = torch.nn.L1Loss(reduction='sum').to(opt.device)


with torch.no_grad():
    loader_val = DataLoader(radio_data_val, batch_size=opt.batchsize) 
    running_loss = 0.

    for idx, data in enumerate(loader_val, 0):
        locs, rss = data
        locs, rss = locs.to(opt.device), rss.to(opt.device)

        rss_pre, weight, Obstacle_maps, rss_calc = radio_net(locs, opt.metainfo.to(opt.device))
        loss = criterion(rss_pre, rss)

        running_loss += loss.item()

        if idx % 10 == 9:
            print('idx: %d, Accumulated MAE: %.3f, Batch MAE: %.3f' % 
            (idx, running_loss/((idx+1)*opt.batchsize), loss.item()/opt.batchsize))