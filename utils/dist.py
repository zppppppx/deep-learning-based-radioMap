import torch
from dataset import locPair
import matplotlib.pyplot as plt

def distance(locs):
    """
    This function realizes the function of computing the log distances between
    the user and the drone. And a number 1 is adjuncted to the tail only for calculation convenience.
    """
    locs1 = locs[:, :3]
    locs2 = locs[:, 3:6]
    dist_absolute = (locs1 - locs2).pow(2).sum(-1).pow(0.5)
    log_dist = torch.log10(dist_absolute).unsqueeze(1)
    ones = torch.ones([locs.size(0), 1])
    return torch.cat((log_dist, ones), dim=1)

def dist(loc):
    """
    This function realizes the function of computing the log distances between
    the user and the drone.
    """
    loc1 = loc[:3]
    loc2 = loc[3:6]
    dist_absolute = (loc1 - loc2).pow(2).sum(-1).pow(0.5)
    log_dist = torch.log10(dist_absolute)

    return log_dist

def dists(locs):
    """
    This function realizes the function of computing the log distances between
    the users and the drones.
    """
    locs1 = locs[:, :3]
    locs2 = locs[:, 3:6]
    dist_absolute = (locs1 - locs2).pow(2).sum(-1).pow(0.5)
    log_dist = torch.log10(dist_absolute)

    return log_dist


if __name__ == '__main__':
    file_path = '../data/radiomap_simulated100tx_3class.mat'
    radio_data_tr = locPair.RadioMap(file_path, fraction=0.0005)

    length = len(radio_data_tr)
    for i in range(length):
        locs, _, rss = radio_data_tr[i]
        log_dist = dist(locs)
