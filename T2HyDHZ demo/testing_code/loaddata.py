import numpy as np
import torch, os
import torch.multiprocessing
from scipy.io import loadmat

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def loadTxt(fn):
    a = []
    with open(fn, 'r') as fp:
        data = fp.readlines()
        for item in data:
            fn = item.strip('\n')
            a.append(fn)
    return a

class dataset_h5(torch.utils.data.Dataset):

    def lmat(self,fn):
        x=loadmat(fn)
        x=x[list(x.keys())[-1]]
        return x

    def __init__(self, X, img_size=256, root='', mode='Train'):
        super(dataset_h5, self).__init__()
        
        self.root = root
        self.fns = X #path
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))
        self.mode=mode
        self.img_size=img_size
 
    def __getitem__(self, index):
        
        fn = os.path.join(self.root, self.fns[index])
        x=loadmat(fn)['hazy_sim']
      
        return x
    
    def __len__(self):

        return self.n_images