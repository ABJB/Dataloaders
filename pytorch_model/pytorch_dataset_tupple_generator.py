import numpy as np
import torch 
from torch.utils import data
import pandas as pd
import glob
import cv2
from DataDownload import DataDownload
import os
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_transform import Transformer


class Dataset_Tuple_Loader(data.Dataset):
    
    
    def __init__(self,img_root, csv_label_address, transforms = None):
        if(device == None):
            # CUDA for PyTorch
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
        self.img_root = img_root
        labels = pd.read_csv(csv_label_address).values
        self.labels = labels
        self.transforms = transforms
        
        
    def __len__(self):
        return len(self.labels)
    
    def get_filename(self,filename):
        base = os.path.basename(filename)
        return (os.path.splitext(base)[0])
    
    def __getitem__(self,index):
        img_name = self.labels[index][0]
        label = self.labels[index][1]
        img = np.asarray(Image.open(self.img_root+ img_name + '.jpeg'))
        if(self.transforms!=None):
            img = self.transforms(img)
        img = torch.tensor(img)
        label  = torch.tensor(label)
        print(type(img))
        return img, label
        





class Dataset(object):
    def __init__(self,data_dict,param_dict):

        ## Download if file is in network
        self.img_source = data_dict['img_source']
        self.img_root = data_dict['img_root']
        self.labels_csv = data_dict['labels_csv']
        self.transforms = Transformer(param_dict['transform_dict']).transforms()
        self.batch_size = param_dict['batch_size']
        self.shuffle = param_dict['shuffle']
        self.num_worker = param_dict['num_worker']
        self.drop_remainder = param_dict['drop_remainder']
        self.device = device


    def __call__(self):
        just_set = Dataset_Tuple_Loader(self.img_root,self.labels_csv,self.transforms)
        params = {'batch_size': self.batch_size,'shuffle': self.shuffle,'num_workers': self.num_worker,'drop_last':self.drop_remainder}
        generator = data.DataLoader(just_set,**params)
        return generator

## Just for testing and future referece purpose
'''
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 1}

csv = '/home/abjb/workspace/facedetection_models/pytorch_retrain/sam.csv'
image_root = '/home/abjb/workspace/facedetection_models/pytorch_retrain/samples'

just_set = Dataset('local',image_root,csv)





for tup in generator:
    img = tup[0]
    lb = tup[1]
    print(img.shape,' ',lb)
'''
