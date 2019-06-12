from torchvision import transforms,utils
import Transformations as tfs
#import numpy as np
#import pandas as pd
#import skimage
#from matplotlib import pylab as plt
#import torch
#import cv2
#from PIL import Image
#import scipy.mis

class Transformer(object):
    def __init__(self, transformation_parameters):   
        
        self.nn_input_image_shape = transformation_parameters['nn_input_image_shape']
        crop_pixel_count = transformation_parameters['crop_pixel']
        self.rescale_shape = [x+y for x,y in zip(self.nn_input_image_shape,crop_pixel_count)]
        self.rotation_angles = transformation_parameters['rotation_angles']
        self.rotation_angle_probabilities = transformation_parameters['rotaion_angle_probabilities']
        self.flip_hor_ver_probabilities = transformation_parameters['flip_hor_ver_probabilities']
        self.gaussian_filter_probability = transformation_parameters['gaussian_filter_probability']
        self.unsharpmask_filter_probability = transformation_parameters['unsharp_mask_filter_probability']
        
    def transforms(self):
        
        composed = transforms.Compose([
                                        tfs.Rescale(out_size=self.rescale_shape),
                                        tfs.ReCrop(out_size =  (self.nn_input_image_shape)),
                                        tfs.Rotate(rotations = self.rotation_angles,rotation_probabilities = self.rotation_angle_probabilities),
                                        tfs.Flip(flip_probabilites = self.flip_hor_ver_probabilities),
                                        tfs.Apply_Gaussian_filter(probabity=self.gaussian_filter_probability),
                                        tfs.Apply_Unsharpmask_filter(probabity=self.unsharpmask_filter_probability)
        ])
        return composed



## Just for testing and future referece purpose
'''

param_dict = {
    'nn_input_image_shape': (100, 100),
    'crop_pixel': (20, 20),
    'rotation_angles' :[25,15,12,0,-12,-15,-25],
    'rotaion_angle_probabilities' :[0.01,0.02,0.12,0.70,0.12,0.02,0.01],
    'flip_hor_ver_probabilities' :(0.4, 0.07),
    'gaussian_filter_probability' : 0.04,
    'unsharp_mask_filter_probability' : 0.04
}
transform = Transformer(param_dict).transforms()

transform(np.asarray(img))
'''

