import Transformations as tfs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

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
        
        self.Rescaler =  tfs.Rescale(out_size=self.rescale_shape)
        self.ReCropper =  tfs.ReCrop(out_size =  (self.nn_input_image_shape))
        self.Rotator = tfs.Rotate(rotations = self.rotation_angles,
                                  rotation_probabilities = self.rotation_angle_probabilities)
        self.Flipper = tfs.Flip(flip_probabilites = self.flip_hor_ver_probabilities)
        self.Gaussian_filter = tfs.Apply_Gaussian_filter(probabity=self.gaussian_filter_probability)
        self.Unsharpmask_filter =  tfs.Apply_Unsharpmask_filter(probabity=self.unsharpmask_filter_probability)
        
    def __call__(self, img):
        img = self.Rescaler(img)
        img = self.ReCropper(img)
        img = self.Rotator(img)
        img = self.Flipper(img)
        img = self.Gaussian_filter(img)
        img = self.Unsharpmask_filter(img)
        return img




### Just for future Reference pupose
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


	transformer = Transformer(param_dict)
	add = '../0samples/face'+ '3.jpeg'

	for i in range(0, 8):
		img = np.asarray(Image.open(add))
		img = transformer(img)
		scipy.misc.imsave('./sampleOut/' + str(i) + '.jpeg',img)	
'''