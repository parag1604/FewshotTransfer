"""
Defines the data generator for generating training data with prior

"""

from medpy.io import save
from medpy.io import load
import numpy as np
import os
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import random_rotation, random_brightness, random_shear, random_shift, random_zoom


def resize(input_image, height, width):
    resized = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return resized

class DataGenerator(Sequence):
    def __init__(self, img_path, mask_path, augmentations, front_buffer, q = 3, batch_size = 5, patch_size = 30, dim = (256,256), shuffle = True, contprop = False, train_with_sparse = False, method = 'Random', percentage = 20, perc_sparse_vol = 100):

        self.img_path = img_path 
        self.mask_path = mask_path
        self.augment = augmentations
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.dim = dim
        self.q = q
        self.shuffle = shuffle
        self.contprop = contprop
        self.method = method
        self.percentage = percentage
        self.perc_sparse_vol = perc_sparse_vol
        
        #Get file names for training
        if(train_with_sparse):
            self.fnames = self.get_slices_for_sparse_training()
        else:
            target_fnames = os.listdir(img_path)
            
            to_remove_fnames = []
            for filename in target_fnames:
              slice_no = int(filename.split('_slice')[1].split('.')[0])
              if(slice_no < front_buffer):
                  to_remove_fnames.append(filename)
            self.fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
        
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fnames) / self.batch_size))
    
    def get_slices_for_sparse_training(self):
        
        #Getting names of files to use for q-CNN
        target_fnames = os.listdir(self.img_path)
    
        #Getting the number and name of volumes
        patients = []
        for file in target_fnames:
            pat_name = file.split('_slice')[0]
            patients.append(int(pat_name))
        patients = np.unique(patients)
        
        #Getting the number of slices corresposnding to each volume
        slices =  [[] for i in range(len(patients))]
        for file in target_fnames:
            pat_name = file.split('_slice')[0]
            slice_no = file.split('_slice')[1].split('.')[0]
            slices[np.where(patients == int(pat_name))[0][0]].append(int(slice_no))
        
        slices = [np.sort(slices[i]) for i in range(len(slices))]
        
        #choosing the volumes with sparse annotations
        index50 = np.random.randint(0, len(patients), size = (int(len(patients)*self.perc_sparse_vol/100)))
        
        #Getting the required slices from each volume
        if(self.method == 'Central'):
            perc_to_remove_each_side = (100 - self.percentage)/2  
            for i in range(len(patients)):
                s = slices[i]
                if(i in index50):
                    l = len(s)
                    num_to_remove_each_side = int(np.ceil(perc_to_remove_each_side*l/100))
                    to_remove = np.concatenate((s[0:num_to_remove_each_side], s[-num_to_remove_each_side:]))
                    to_remove_fnames = [[name for name in target_fnames if name.startswith(str(patients[i]) + '_slice' + str(x) + '.')][0] for x in to_remove]
                    target_fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
                else:
                    to_remove = range(0,self.q)
                    to_remove_fnames = [[name for name in target_fnames if name.startswith(str(patients[i]) + '_slice' + str(x) + '.')][0] for x in to_remove]
                    target_fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
        elif(self.method == 'Front'):
            for i in range(len(patients)):
                s = slices[i]
                to_remove = range(0,self.q)
                to_remove_fnames = [[name for name in target_fnames if name.startswith(str(patients[i]) + '_slice' + str(x) + '.')][0] for x in to_remove]
                target_fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
                if(i in index50):
                    l = len(s)
                    front_num = int(np.floor((self.percentage/100)*l)) + self.q
                    to_remove = range(front_num, l)
                    to_remove_fnames = [[name for name in target_fnames if name.startswith(str(patients[i]) + '_slice' + str(x) + '.')][0] for x in to_remove]
                    target_fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
        elif(self.method == 'Random'):
            for i in range(len(patients)):
                s = slices[i]
                to_remove = range(0,self.q)
                to_remove_fnames = [[name for name in target_fnames if name.startswith(str(patients[i]) + '_slice' + str(x) + '.')][0] for x in to_remove]
                target_fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
                if(i in index50):
                    l = len(s)
                    num = int(np.floor((self.percentage/100)*l)) 
                    start = np.random.randint(self.q, l - num)
                    to_remove = np.concatenate((np.arange(self.q,start), np.arange(start + num, l)))
                    to_remove_fnames = [[name for name in target_fnames if name.startswith(str(patients[i]) + '_slice' + str(x) + '.')][0] for x in to_remove]
                    target_fnames = np.array([name for name in target_fnames if name not in to_remove_fnames])
        
        return target_fnames
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.fnames)

    def __getitem__(self, index):
        'Generate one batch of data'
        data_index_min = int(index * self.batch_size)
        data_index_max = int((index + 1) * self.batch_size)

        batch_fnames = self.fnames[data_index_min:data_index_max]
        #Generate Data
        img, gt, Y = self.__data_generation(batch_fnames)
        
        if(self.contprop):
            X = np.empty((self.batch_size, *self.dim, 2))
            Y = np.empty((self.batch_size, *self.dim, 1))
            #Augment Data
            for i in np.arange(self.batch_size):
                augmented = self.augment(image = img[i,:,:,:].astype('float32'), mask = gt[i,:,:,:].astype('float32'))
                X[i,:,:,0:1] = augmented["image"]
                X[i,:,:,1:] = augmented["mask"][:,:,1:]
                Y[i] = augmented["mask"][:,:,0:1]
            
            return X, Y
        
        else:
            Ximg = np.empty((self.batch_size, *self.dim, 1))
            Xgt = np.empty((self.batch_size, *self.dim, 1))
            Xprior = np.empty((self.batch_size, *self.dim, 1))
    
            #Augment Data
            for i in np.arange(self.batch_size):
                augmented = self.augment(image = img[i,:,:,:].astype('float32'), mask = gt[i,:,:,:].astype('float32'))
                Ximg[i] = augmented["image"]
                Xgt[i] = augmented["mask"][:,:,0:1]
                Xprior[i] = augmented["mask"][:,:,1:]

            return Ximg, Xgt, Xprior, Y, batch_fnames


    def __data_generation(self, batch_fnames):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        Ximg = np.empty((len(batch_fnames), *self.dim, 1))
        Xgt = np.empty((len(batch_fnames), *self.dim, 2))
        Y = np.full((len(batch_fnames), self.patch_size, self.patch_size, 1), tf.random.uniform(shape = [], minval = 0.9, maxval = 1))

        # Generate data
        for i in np.arange(len(batch_fnames)):
             
            img, target_header = load(self.img_path + batch_fnames[i])

            
            slice_no = batch_fnames[i].split('_slice')[1].split('.')[0]
            pat_name = batch_fnames[i].split('_slice')[0]
            
            prior_pos = int(slice_no) - self.q
            prior, prior_header = load(self.mask_path + pat_name + '_slice' + str(prior_pos) + '.nii.gz')
            prior[prior > 0] = 1

            if(self.mask_path is not False):
                mask_img, mask_header = load(self.mask_path + batch_fnames[i])
                mask_img[mask_img > 0]= 1
            else:
                mask_img = np.zeros_like(img)
                
            stack = tf.stack([img, mask_img, prior], axis = -1)
            resized = resize(stack, self.dim[0], self.dim[1])
            Ximg[i] = resized[:,:,0:1]
            Xgt[i,:,:,0] = resized[:,:,1]
            Xgt[i,:,:,1] = resized[:,:,2]

        return Ximg, Xgt, Y

    def data_gen_real(self):
        
        index = 0
        batch = 1
        n_batches_per_epoch = self.__len__()
        self.on_epoch_end()
    
        while(True):
            
            if(self.contprop):
                X, Y = self.__getitem__(index)
            else:
                [Ximg, Xgt, Xprior, Y, batch_fnames] = self.__getitem__(index)
            
            index += 1
            batch += 1
            
            if(index == n_batches_per_epoch):
                batch = 1
                index = 0
                self.on_epoch_end()
                
            if(self.contprop):
                yield X,Y
            else:                
                yield Ximg, Xprior, [Xgt, Y], batch_fnames