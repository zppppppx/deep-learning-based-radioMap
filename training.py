import os
import numpy as np
import time
import torch
from dataset import locPair
from modules import algorithmSim, weightGen
import utils.Config as Config
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.load_dict import load_dict

##################
# Basic Settings #
##################
opt = Config.Config()
file_path = '../data/radiomap_simulated100tx_3class_noise.mat'
# file_path = '../data/radiomap_simulated100tx.mat'
net_path = opt.RE_Net + '_3_' + str(opt.map_order+1) + '.pth' # 3 for 3class. Suit yourself.


weightNet_ob_path = './weight_ob_' + str(opt.map_order) + '.pth'
weightNet_rss_path = './weight_rss_' + str(opt.map_order) + '.pth'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
radio_data_tr = locPair.RadioMap(file_path, choice='synthetic')
batchsize = opt.batchsize
epoch = opt.epoch

weight_rss = weightGen.weightGen_rss(opt).to(device)
radio_net = algorithmSim.Algorithm(opt).to(device) # RadioNet

# Load trained part
radio_net = load_dict(radio_net, weightNet_ob_path, 'weight_gen_ob')
weight_rss.load_state_dict(torch.load(weightNet_rss_path))


#################
# Training Loss #
#################
criterion_rss = torch.nn.MSELoss().to(device)
criterion_ob = torch.nn.MSELoss().to(device)


#############
# Optimizer #
#############
optimizer_radio_RENet = optim.Adam([{'params':radio_net.params_model.parameters(), 'lr': 0},
                        {'params':radio_net.map.parameters(), 'lr': opt.lr_map},
                        {'params':radio_net.weight_gen_ob.parameters(), 'lr': 0}])

optimizer_radio_multiChannel = optim.SGD([{'params':radio_net.params_model.parameters(), 'lr': opt.lr_param},
                        {'params':radio_net.map.parameters(), 'lr': 0},
                        {'params':radio_net.weight_gen_ob.parameters(), 'lr': 0}])

optimizer_radio_finetune = optim.Adam([{'params':radio_net.params_model.parameters(), 'lr': 5e-3},
                    {'params':radio_net.map.parameters(), 'lr': 5e-3},
                    {'params':radio_net.weight_gen_ob.parameters(), 'lr': 0}])


####################
# Training section #
####################
T = True
F = False
train_rss = T
train_ob = T
finetune = F


if os.path.exists(net_path):
        state = torch.load(net_path)
        radio_net.load_state_dict(state['net'])
print(radio_net.params_model.state_dict())

time_start=time.time()

for epoch in range(opt.epoch):
    ######## Obstacle map
    if train_ob:
        for i in range(opt.epoch_map):
            loader_tr = DataLoader(radio_data_tr, batch_size=batchsize, shuffle=True)

            running_loss = 0.

            for idx, data in enumerate(loader_tr, 0):
                locs, rss = data
                locs = locs.to(device)
                rss = rss.to(device)

                rss_pre, weight, Obstacle_maps, rss_calc = radio_net(locs, opt.metainfo.to(device))
                std = weight_rss(rss_calc, rss.unsqueeze(1))

                loss = 10*criterion_ob(weight, std) # MSE loss
                loss.backward()
                optimizer_radio_RENet.step()
                optimizer_radio_RENet.zero_grad()
                running_loss += loss.item()


                if idx % 10 == 9:
                    print('RE-Net: outer: %d, inner: %d, idx: %d, loss: %.4f'%(epoch, i, idx, running_loss/10))
                    # print(rss_calc[:3], rss[:3], classes[:3], weight[:3])
                    # print(radio_net.map.outcome.state_dict())
                    # plt.imshow(Obstacle_maps.cpu().detach().squeeze(), cmap='plasma', vmin=0, vmax=2)
                    # plt.show()
                    state = {'net': radio_net.state_dict()}
                    torch.save(state, net_path)

                    running_loss = 0.
                    # if idx == 169:
                    #     break

        state = {'net': radio_net.state_dict()}
        torch.save(state, net_path)


    ############ RSS
    if train_rss:
        for i in range(opt.epoch_param):
            loader_tr = DataLoader(radio_data_tr, batch_size=batchsize, shuffle=True)

            running_loss = 0.
            # check = torch.ones([2,2]).cuda()
            for idx, data in enumerate(loader_tr, 0):
                locs, rss = data
                locs = locs.to(device)
                rss = rss.to(device)

                rss_pre, weight, Obstacle_maps, rss_calc = radio_net(locs, opt.metainfo.to(device))
                

                loss = criterion_rss(rss_pre, rss)
                loss.backward()
                optimizer_radio_multiChannel.step()
                optimizer_radio_multiChannel.zero_grad()
                running_loss += loss.item()


                if idx % 10 == 9:
                    print('Multi-Channel: outer: %d, inner: %d, idx: %d, loss: %.4f'%(epoch, i, idx, running_loss/10))
                    print(radio_net.params_model.state_dict())
                    state = {'net': radio_net.state_dict()}
                    torch.save(state, net_path)

                    running_loss = 0.

                    # if idx == 169:
                    #     break

        state = {'net': radio_net.state_dict()}
        torch.save(state, net_path)

    time_end=time.time()
    print('Time consumed: ', time_end-time_start)



if finetune:
    for i in range(opt.epoch_finetune):
        loader_tr = DataLoader(radio_data_tr, batch_size=batchsize, shuffle=True)

        running_loss = 0.
        # check = torch.ones([2,2]).cuda()
        for idx, data in enumerate(loader_tr, 0):
            locs, rss = data
            locs = locs.to(device)
            rss = rss.to(device)

            rss_pre, weight, Obstacle_maps, rss_calc = radio_net(locs, opt.metainfo.to(device))
            std = weight_rss(rss_calc, rss.unsqueeze(1))


            loss = criterion_rss(rss_pre, rss)
            loss.backward()
            optimizer_radio_finetune.step()
            optimizer_radio_finetune.zero_grad()
            running_loss += loss.item()


            if idx % 2 == 1:
                print('Mini-modu: whole: %d, epoch: %d, idx: %d, loss: %.4f'%(epoch, i, idx, running_loss/10))
                print(radio_net.params_model.state_dict())
                state = {'net': radio_net.state_dict()}
                torch.save(state, net_path)

                running_loss = 0.

                # if idx == 99:
                #     break

state = {'net': radio_net.state_dict()}
torch.save(state, net_path)
