import numpy as np
import torch
from torch import tensor as t
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import Map
from utils import dist

# import sys
# sys.path.append('./utils')
# import Map

class RadioMap(Dataset):
    def __init__(self, file_path, choice='synthetic', fraction=None, seed=1):
        All_data = loadmat(file_path)
        self.radio_map = All_data['RadioMap']
        self.length = self.radio_map.shape[0]
        # self.marks = [int(marks[0]*self.length), int(marks[1]*self.length)]
        np.random.seed(seed)
        
        if fraction == None:
            fraction = 10000./self.length
        rand_index = np.random.randint(0,self.length, int(fraction*self.length))
        self.locs = self.radio_map[rand_index, :6]#self.radio_map[self.marks[0]:self.marks[1], :6]

        # Let the ordinate begin from 0.
        if choice == 'synthetic':
            self.locs[:, np.array([0,1,3,4])] -= 3
        if choice == 'simulated':
            self.locs[:, np.array([0,1,3,4])] += np.array([26.71, 131.3, 26.71, 131.3])

        # self.Position_maps = self.locs_to_maps()
        self.rss = self.radio_map[rand_index, -1] #[self.marks[0]:self.marks[1], -1]


    def __len__(self):
        return self.rss.shape[0]

    def __getitem__(self, index):
        return  t(self.locs[index]).float(), t(self.rss[index]).float()
        


    


if __name__ == '__main__':
    file_path = '../data/radiomap_simulated100tx_3class.mat'
    data = loadmat(file_path)
    print(data['RadioMap'].shape)
    print(Map.loc_pair_to_map())