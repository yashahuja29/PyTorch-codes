import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np

class Wine(Dataset):

    def __init__(self):
            xy = np.loadtxt("wine.csv",delimiter=",",dtype=np.float32,skiprows=1 )
            self.x = torch.from_numpy(xy[:,1:])
            self.y = torch.from_numpy(xy[:,[0]])
            self.n_sample = xy.shape[0]


    def __getitem__(self,idx):
          return self.x[idx],self.y[idx]
          

    
    def __len__(self):
          return self.n_sample

data = Wine()
datal = DataLoader(dataset=data, batch_size=4,shuffle=True)


#training
num_epoch = 2
total_samp = len(data)
n_iter = np.ceil(total_samp/4)
print(total_samp,n_iter)

for epoch in range(num_epoch):
      for i, (input,label) in enumerate(datal):
            #forward pass
            if (i+1)%5==0:
                  print()

