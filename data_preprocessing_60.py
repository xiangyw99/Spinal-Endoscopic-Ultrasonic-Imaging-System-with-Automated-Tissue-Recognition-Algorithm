import torch
import monai.transforms as mt
import numpy as np
from glob import glob
import sys
sys.path.append('./')
import torch

import numpy as np

import os
import monai.transforms as mt

from utils import threshold_at_one, jpg_to_ndarray


class ABUSsetNew(torch.utils.data.Dataset):
    def __init__(self, args, fold=None, mode='train'):
        if args.stage==0:
            self.root=args.imagesTr+f"/fold_{fold}/{mode}"
        self.mode=mode
        files=sorted(glob(self.root+"/*.npy"))
        self.label=[]
        self.label_stage1=[]
        self.label_stage2=[]
        self.image_path=[]

        for file in files:
            n,name=os.path.split(file)
            c,_,_=name.split("_")
            if c == 'jisui':
                
                self.image_path.append(file)
                self.label_stage1.append(0)
                self.label_stage2.append(0)
                self.label.append(0)
            elif c == 'suihe':
                
                self.image_path.append(file)
                self.label_stage1.append(0)
                self.label_stage2.append(1)
                self.label.append(1)
            elif c == 'zhifang':
                
                self.image_path.append(file)
                self.label_stage1.append(0)
                self.label_stage2.append(2)
                self.label.append(2)
            elif c =='gutou':
                
                self.image_path.append(file)
                self.label_stage1.append(1)
                self.label_stage2.append(0)
                self.label.append(3)
            elif c=='xianweihuan':
              
                self.image_path.append(file)
                self.label_stage1.append(1)
                self.label_stage2.append(1)
                self.label.append(4)
            elif c=='shenjinggen':
               
                self.image_path.append(file)
                self.label_stage1.append(2)
                self.label_stage2.append(0)
                self.label.append(5)
        assert len(self.label)==len(self.image_path)
        assert len(self.image_path)!=0
        print("total dataset length: ", len(self.image_path))

        self.trans=self.get_transform()
    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image=np.load(self.image_path[idx])

        label=self.label[idx]
        label_stage1=self.label_stage1[idx]
        label_stage2=self.label_stage2[idx]
        image=self.trans(image)

        _,name=os.path.split(self.image_path[idx])
        return {'image':image, 'label':torch.tensor(label).long(), 'label_stage1': torch.tensor(label_stage1).long(), 'label_stage2': torch.tensor(label_stage2).long(), 'name':name}
    
    def get_transform(self):
        if self.mode=='train':
            transform = mt.Compose([
                mt.CropForeground(select_fn=threshold_at_one, margin=0),
                
                mt.EnsureChannelFirst(channel_dim='no_channel'),
                mt.RandGaussianNoise(mean=0, std=1),
                mt.RandFlip(prob=0.3),
                mt.RandZoom(prob=0.25, min_zoom=0.6, max_zoom=0.8),
                mt.RandAffine(prob=0.3, shear_range=(0.5,0.5)),
                
                mt.RandAdjustContrast(prob=0.3, gamma=(1.5,2)),
                mt.Resize(spatial_size=(224,224)),
            ])
        elif self.mode=='val':
            transform = mt.Compose([
                mt.CropForeground(select_fn=threshold_at_one, margin=0),
                mt.EnsureChannelFirst(channel_dim='no_channel'),
                mt.Resize(spatial_size=(224, 224)),
                mt.NormalizeIntensity(subtrahend=0, divisor=1)
            ])
        return transform

if __name__ == '__main__':
    pass