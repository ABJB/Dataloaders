from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

#tf.enable_eager_execution()

from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import misc
from generic_Transformer import Transformer



class Generator(object):
    def __init__(self,img_root,csv_label_address,transforms):
        self.img_root = img_root
        label_np = pd.read_csv(csv_label_address).values
        self.labels = label_np
        self.transforms = transforms

    def gen(self):
        for tup in self.labels:
            img = np.asarray(Image.open(self.img_root + tup[0] + '.jpeg'))
            label = tup[1]
            img = self.transforms(img)
            yield {'imgs':img,'labels':label}
        


    def get_filename(self,filename):
        base = os.path.basename(filename)
        return (os.path.splitext(base)[0])

    def __call__(self, batch_size,shuffle,num_worker,drop_remainder):
        set = tf.data.Dataset.from_generator(self.gen, output_types={'imgs':tf.float32,'labels':tf.int32})
        generator = tf.data.Dataset.batch(set,batch_size,drop_remainder = drop_remainder)
        return generator






class Dataset(object):
    def __init__(self,data_dict,param_dict):

        ## Download if file is in network
        self.img_source = data_dict['img_source']
        self.img_root = data_dict['img_root']
        self.labels_csv = data_dict['labels_csv']
        self.transforms = Transformer(param_dict['transform_dict'])
        self.batch_size = param_dict['batch_size']
        self.shuffle = param_dict['shuffle']
        self.num_worker = param_dict['num_worker']
        self.drop_remainder = param_dict['drop_remainder']

    def __call__(self):
        generator = Generator(self.img_root,self.labels_csv,self.transforms)(self.batch_size,self.shuffle,self.num_worker,self.drop_remainder)
        return generator


##Just For Ref

