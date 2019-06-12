import cv2
from  pytorch_dataset_tupple_generator import Dataset
import scipy.misc
import torch

transform_dict = {
        'nn_input_image_shape': (100, 100),
        'nn_input_channels' : 3,
        'crop_pixel': (20, 20),
        'rotation_angles' :[25,15,12,0,-12,-15,-25],
        'rotaion_angle_probabilities' :[0.01,0.02,0.12,0.70,0.12,0.02,0.01],
        'flip_hor_ver_probabilities' :(0.4, 0.07),
        'gaussian_filter_probability' : 0.04,
        'unsharp_mask_filter_probability' : 0.04
    }
param_dict = {
    'transform_dict': transform_dict,
    'batch_size':3,
    'num_worker':1,
    'drop_remainder': True,
    'shuffle':True
}

data_dict = {
    'img_source' : 'local',
    'img_root': './sample_images/',
    'labels_csv': './sam.csv',
    'num_classes' : 8
}



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")




data = Dataset(data_dict,param_dict,device)()

for imgs,labels in data:
    print(imgs[0].shape)
    scipy.misc.toimage(imgs[0]).save('./sample_out/' + str(labels[0])+'.jpeg')
    scipy.misc.toimage(imgs[1]).save('./sample_out/' + str(labels[1])+'.jpeg')
