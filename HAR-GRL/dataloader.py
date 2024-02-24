import torch 
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class CSIDataset(Dataset):
    def __init__(self, metadata):
        super().__init__()
        self.csiFiles, self.targets, self.domains= self.load_dataset(metadata)
    
    def __len__(self):
        return self.csiFiles.shape[0]
    
    def __getitem__(self, index):
        csi = np.load(self.csiFiles[index], allow_pickle=True)
  
        csi_real = np.abs(csi) 
        csi_imag = np.angle(csi)
        
        csi = np.swapaxes(csi_real, 0, -1)
        csi = torch.from_numpy(csi)

        label = self.targets[index]
        target = torch.tensor(label)
        
        domain = self.domains[index]
        domain = torch.tensor(domain)

        return csi, target, domain

    def load_dataset(self, metadata):
        csiFiles = []
        targets = []
        domains = []

        with open(metadata, 'r') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split('\n')[0].split(' ')

                file = str(words[0])
                target = int(words[1])
                env = int(words[2])

                csiFiles.append(file)
                targets.append(target)
                domains.append(env)
            
        f.close()
      
        csiFiles = np.array(csiFiles)
        targets = np.array(targets)
        domains = np.array(domains)

        return csiFiles, targets, domains
