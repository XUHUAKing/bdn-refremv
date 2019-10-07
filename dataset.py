import os
import itertools
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from glob import glob
import random
import scipy.misc


class ref_dataset(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 rf_transform=None,
                 real=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.rf_transform = rf_transform
        self.real = real
        if real:
            self.ids = []
            img_names = glob(root+'/*.npy')
            img_names.sort()
            mask_names = glob(root+'/mask' + '/*mask.png')
            mask_names.sort()
            for tmp_mask in mask_names:
                tmp_M = root + '/IMG_' + tmp_mask[-18:-14] + '.npy'
                tmp_R = root + '/IMG_' + tmp_mask[-13:-9] + '.npy'
                if os.path.isfile(tmp_M) and os.path.isfile(tmp_R):
                    self.ids.append([tmp_M, tmp_mask])
                else:
                    print(tmp_M, tmp_R, tmp_mask, 'not exist...')
                    raise Exception('M/R/Mask not exist')
            print("Data load succeed!")
        else:
            self.ids = sorted(os.listdir(os.path.join(root, 'I')))

    def __getitem__(self, index):
        img, mask= self.ids[index]
        if self.real:
            # input = Image.open(os.path.join(self.root, img)).convert('RGB')
            input = np.load(img)[0,:,:,-1]# (1024, 1224) H*W
            tmp_mask=scipy.misc.imread(mask,'L')[::2,::2,np.newaxis]/255.
            input = input[:,:,np.newaxis]*tmp_mask # crop valid region
            input = np.tile(input,[1,1,3])
            if self.transform is not None:
                input = self.transform(input)
            return input
        else:
            input = Image.open(os.path.join(self.root, 'I', img)).convert('RGB')
            target = Image.open(os.path.join(self.root, 'B', img)).convert('RGB')
            target_rf = Image.open(os.path.join(self.root, 'R', img)).convert('RGB')
            if self.transform is not None:
                input = self.transform(input)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.rf_transform is not None:
                target_rf = self.rf_transform(target_rf)
            return input, target, target_rf

    def __len__(self):
        return len(self.ids)
