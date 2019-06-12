import numpy as np
import keras
import pandas as pd
from PIL import Image
from generic_Transformer import Transformer

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,img_root,csv_labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True,transforms = None):
        'Initialization'
        self.img_root = img_root
        label_np = pd.read_csv(csv_labels).values
        self.labels = label_np
        self.transforms = transforms

        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i,ID in enumerate(list_IDs_temp):
            # Store sample
            img_name,label = self.labels[ID] 
            img  = np.asarray(Image.open(self.img_root + img_name + '.jpeg'))
            if(self.transforms!=None):
                img = self.transforms(img)
            X[i,] = img
            # Store class
            y[i] = label

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



class Dataset(object):
    def __init__(self,data_dict,param_dict):

        ## Download if file is in network
        self.labels_csv = data_dict['labels_csv']
        self.img_source = data_dict['img_source']
        self.img_root = data_dict['img_root']
        self.transforms = Transformer(param_dict['transform_dict'])
        self.batch_size = param_dict['batch_size']
        self.shuffle = param_dict['shuffle']
        self.num_worker = param_dict['num_worker']
        self.drop_remainder = param_dict['drop_remainder']
        self.out_image_shape = param_dict['transform_dict']['nn_input_image_shape']
        self.out_channels = param_dict['transform_dict']['nn_input_channels']
        self.num_classes = data_dict['num_classes']

    def __call__(self):
        generator = DataGenerator(self.img_root,self.labels_csv,
                                self.batch_size,self.out_image_shape,
                                self.out_channels,self.num_classes,self.shuffle,self.transforms)
        return generator
