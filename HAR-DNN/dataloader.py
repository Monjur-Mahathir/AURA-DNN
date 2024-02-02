import torch 
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class CSIDataset(Dataset):
    
    def __init__(self, metadata):
        super().__init__()
        self.csiFiles, self.targets= self.load_dataset(metadata)
    
    def __len__(self):
        return self.csiFiles.shape[0]
    
    def __getitem__(self, index):
        csi = np.load(self.csiFiles[index], allow_pickle=True)
  
        csi_real = np.abs(csi) 
        #csi_imag = np.angle(csi)
        #csi = np.concatenate([csi_real, csi_imag], axis=-1)
        #csi = np.swapaxes(csi, 0, -1)
        
        csi = np.swapaxes(csi_real, 0, -1)
        csi = torch.from_numpy(csi)

        label = self.targets[index]
        target = torch.tensor(label)

        return csi, target

    def load_dataset(self, metadata):
        csiFiles = []
        targets = []

        with open(metadata, 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split('\n')[0].split(' ')

                file = str(words[0])
                target = int(words[1])
                env = int(words[2])
                if target == -1:
                    continue

                csiFiles.append(file)
                targets.append(target)

        f.close()
        csiFiles = np.array(csiFiles)
        targets = np.array(targets)

        return csiFiles, targets
