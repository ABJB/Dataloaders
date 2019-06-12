import skimage
import numpy as np
from skimage import transform

class Rescale(object):
    def __init__(self, out_size):
        assert isinstance(out_size,(int, tuple, list))
        if(isinstance(out_size,int)):
            out_size = (out_size, out_size)
        assert len(out_size) == 2
        self.out_size = out_size
    
    def __call__(self, img):        
        new_h,new_w,_ =h,w,__ = img.shape        
        if(isinstance(self.out_size,int)):
            if(h>w):
                new_h,new_w = self.out_size*h/w, self.out_size
            else:
                new_h,new_w = self.out_size,self.out_size*w/h
        else:
            new_h,new_w = self.out_size        
        new_h,new_w = int(new_h), int(new_w)        
        img = transform.resize(img, (new_h,new_w))
    
        return img


class ReCrop(object):
    def __init__(self, out_size):
        assert isinstance(out_size, (int, tuple,list))
        if(isinstance(out_size,int)):
            out_size = (out_size, out_size)
        assert len(out_size) == 2
        self.out_size = out_size
        
    def __call__(self, img):        
        h,w,_ = img.shape
        new_h,new_w = self.out_size
        ver_start = np.random.randint(0,h - new_h)
        hor_start = np.random.randint(0,w - new_w)
        img = img[ver_start:ver_start+new_h, hor_start:hor_start+new_w]
        return img


class Rotate(object):
    def __init__(self,rotations = [25,15,12,0,-12,-15,-25], rotation_probabilities = [0.01,0.02,0.12,0.70,0.12,0.02,0.01]):
        '''	
            parameters : 
                rotations : all possible rotations to be applied.
                rotation_probabilities : choosing rotation with given probabilities.
        '''
        assert isinstance(rotation_probabilities,(list, tuple)) and isinstance(rotations,(list,tuple)) and len(rotation_probabilities)==len(rotations) and np.sum(np.array(rotation_probabilities))==1.00 
        self.rotation_probabilities =rotation_probabilities
        self.rotations = rotations
    
    def __call__(self, img):        
        angle = np.random.choice(a = self.rotations,p = self.rotation_probabilities)
        img = transform.rotate(angle=angle,image=img,mode='edge')
        return img





class Flip(object):
    def __init__(self, flip_probabilites = (0.4, 0.07)):
        '''
            flip_probabilites : 2 element list or array with probability of horizontal and vertical flip respectively
        '''
        assert isinstance(flip_probabilites,(list, tuple)) and len(flip_probabilites) == 2 and all(x !=4 for x in flip_probabilites)
        self.flip_probabilites = flip_probabilites
    
    def __call__(self, img):        
        if((np.random.choice(a = [1, 0],p = [self.flip_probabilites[0], 1 -self.flip_probabilites[0]])) == 1):
            img = img[:, ::-1]
        if((np.random.choice(a = [1, 0],p = [self.flip_probabilites[1], 1 -self.flip_probabilites[1]])) == 1):
            img = img[::-1,:]
        return img



class Apply_Gaussian_filter(object):
    def __init__(self, probabity=0.04):
        assert probabity<=1.0
        self.probabity = probabity

    def __call__(self,img):
        img = skimage.filters.gaussian(image = img, multichannel=True)
        return img
    
class Apply_Unsharpmask_filter(object):
    def __init__(self, probabity=0.04):
        assert probabity<=1.0
        self.probabity = probabity

    def __call__(self,img):
        img = skimage.filters.unsharp_mask(image = img)
        return img