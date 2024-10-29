import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from loaddata import dataset_h5, loadTxt
import scipy.io
from networks.T2HyDHZ import IPT_dehazenet

def test(model,val_gen):
    
    model.eval()

    with torch.no_grad():

        for iteration, (Hazy) in enumerate(val_gen):
    
            Hazy=np.transpose(Hazy,(0,3,1,2)).float().to('cuda')
            warmup_output,_ = model(torch.randn(1, 172, 256, 256).to('cuda'))          
            start_time = time.time()
            dehaze,_= model(Hazy)
            time_dl = time.time()-start_time          
            result3d=dehaze[0].permute(1,2,0).cpu().numpy()           
            scipy.io.savemat("./testing_code/results/XDL.mat" ,{'XDL':result3d,'time_dl': time_dl})  
   
    return dehaze

flist = loadTxt("./testing_code/test_haze.txt")
val_set = dataset_h5(flist, root='')
valloader = DataLoader(val_set, batch_size=1)
model=IPT_dehazenet().to("cuda")
model.load_state_dict(torch.load('./testing_code/model_checkpoint.pth'))
result = test(model,valloader)