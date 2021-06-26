# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:18:51 2020

@author: Mehak

"""

#Allow GPU Growth
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)

#Import Libraries
import numpy as np 
import os
import cv2
import time
from tensorflow.keras.utils import Progbar
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomBrightness, RandomContrast,
    ToFloat, ShiftScaleRotate, Resize
)


#Import Modules
from models import discriminator, unet, unet_withBFT, guided_gan
from utils import dice_coef, generate_fake_samples, dice_loss, vgg_loss, Cut_VGG19
from datagen import DataGenerator

#Arguments
def get_args():
    parser = ArgumentParser(description = 'Pix2Pix_ArgumentParser')
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_path_top', type=str, default = './Out/')
    parser.add_argument('--save_path_add', type=str)
    parser.add_argument('--data_path', type=str, default = './ctdata/Axial/newslices/Lumbar/')
    parser.add_argument('--val_path', type=str, default = './ctdata/Axial/newslices/Lumbar/Val/')
    parser.add_argument('--load_pretrained', type=bool, default = False)
    parser.add_argument('--pretrained_disc_weight_path', type=str, default = '.')
    parser.add_argument('--pretrained_gen_weight_path', type=str, default = '.')
    parser.add_argument('--pretrained_cpp_weight_path', type=str, default = '.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lrg', type=float, default=0.0002)
    parser.add_argument('--lrd', type=float, default=0.0002)
    parser.add_argument('--lrcpp', type=float, default=0.0001)
    parser.add_argument('--resize_height', type=int, default=286)
    parser.add_argument('--resize_width', type=int, default=286)
    parser.add_argument('--crop_height', type=int, default=256)
    parser.add_argument('--crop_width', type=int, default=256)
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--front_buffer', type=int, default=10)
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--segmentation_loss', type=str, default='mae')
    parser.add_argument('--discriminator_loss', type=str, default='binary_crossentropy')
    parser.add_argument('--image_guide', type=bool, default=False)
    parser.add_argument('--mask_guide', type=bool, default=False)
    parser.add_argument('--q', type=int, default=3)
    parser.add_argument('--train_with_sparse', type=bool, default = False)
    parser.add_argument('--method', type=str, default="Random")
    parser.add_argument('--percentage', type=int, default=20)
    parser.add_argument('--perc_sparse_vol', type=int, default = 100)

    args = parser.parse_args()
    return args

#Defines and initialises data loaders 
def get_data(args, contprop = False, patch_out=30):
    
    DATA_PATH_SRC = args.data_path    
    
    #Slices to be stored in folders named "Image" and ground truth in folder named "Mask"
    IMG_PATH_SRC = os.path.join(DATA_PATH_SRC + 'Image/') 
    MASK_PATH_SRC = os.path.join(DATA_PATH_SRC + 'Mask/')

    BATCH_SIZE = args.batch_size
    INPUT_DIM = [args.crop_height, args.crop_width, 1]
    q = args.q
    FRONT_BUFFER = args.front_buffer
    train_with_sparse = args.train_with_sparse
    method = args.method
    percentage = args.percentage
    perc_sparse_vol = args.perc_sparse_vol

    AUGMENTATIONS_SOURCE = Compose([
    HorizontalFlip(p=0.5), 
    VerticalFlip(p = 0.5),
    ShiftScaleRotate(
        shift_limit=0.1, scale_limit=(0.75,1.25), 
        rotate_limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.5)
    ])
    
    #Initialising the data loader for training
    params_train = {
            'img_path': IMG_PATH_SRC,
            'mask_path': MASK_PATH_SRC,
            'augmentations': AUGMENTATIONS_SOURCE,
            'batch_size': BATCH_SIZE,
            'q': q,
            'patch_size': patch_out,
            'dim': (INPUT_DIM[0], INPUT_DIM[1]),
            'front_buffer': FRONT_BUFFER,
            'shuffle': True,
            'contprop': contprop,
            'train_with_sparse': train_with_sparse,
            'method' : method,
            'percentage' : percentage,
            'perc_sparse_vol': perc_sparse_vol
        }
        
    train_data = DataGenerator(**params_train)
        
    if(args.validation):
       
        DATA_PATH_VAL = args.val_path
        IMG_PATH_VAL = os.path.join(DATA_PATH_VAL + 'Image/') 
        MASK_PATH_VAL = os.path.join(DATA_PATH_VAL + 'Mask/')
        AUGMENTATIONS_VAL = Compose([])
        
        params_val = {
        'img_path': IMG_PATH_VAL,
        'mask_path': MASK_PATH_VAL,
        'augmentations': AUGMENTATIONS_VAL,
        'batch_size': BATCH_SIZE,
        'q': q,
        'patch_size': patch_out,
        'dim': (INPUT_DIM[0], INPUT_DIM[1]),
        'front_buffer': FRONT_BUFFER,
        'shuffle': False,
        'contprop': contprop,
        'train_with_sparse': False
         }
        
        val_data = DataGenerator(**params_val)
        
        return train_data, val_data
    else:
        return train_data

    
def train_guided_pix2pix(WEIGHTS_PATH, RESULTS_PATH, args):

    ### Define Parameters
    EPOCHS = args.epochs
    INPUT_DIM = [args.crop_height, args.crop_width, 1]
    lrd = args.lrd
    lrg = args.lrg
    func_dict = {
            'dice_loss' : dice_loss,
            'mae' : 'mae',
            'vgg_loss' : vgg_loss,
            'binary_crossentropy': 'binary_crossentropy'
            }
    
    seg_loss = func_dict[args.segmentation_loss]
    disc_loss = func_dict[args.discriminator_loss]
        
    #Discriminator
    d_model = discriminator(INPUT_DIM)
    d_opt = Adam(lr=lrd, beta_1=0.5)
    d_model.compile(loss=[disc_loss], 
                    metrics= ['accuracy'], 
                    optimizer=d_opt, loss_weights=[0.5])
    
    #Generator
    g_model = unet_withBFT(INPUT_DIM, activation_fn = args.activation)
    
    if(args.load_pretrained):
        g_model.load_weights(args.pretrained_gen_weight_path)
        d_model.load_weights(args.pretrained_disc_weight_path)

    #Guided GAN
    gan_model = guided_gan(g_model = g_model, d_model = d_model, 
                           image_shape = INPUT_DIM,
                           image_guide = args.image_guide, mask_guide = args.mask_guide)
    opt = Adam(lr=lrg, beta_1=0.5)
    gan_model.compile(loss=[disc_loss, seg_loss], 
                      metrics= ['accuracy', dice_coef], 
                      optimizer=opt, loss_weights=[1,100])   
    
    
    #Get data generators
    if(args.validation):
        train_dataloader, val_dataloader = get_data(args, contprop = False, 
                                                    patch_out = d_model.output_shape[1])
        train_gen = train_dataloader.data_gen_real()
        val_gen = val_dataloader.data_gen_real()
    else: 
        train_dataloader = get_data(args, contprop = False, 
                                    patch_out = d_model.output_shape[1])
        train_gen = train_dataloader.data_gen_real()
        
    #Log Values
    with open(RESULTS_PATH + 'logtrain.txt', 'a') as log_train:
            log_train.writelines(["Epoch",";","d_real_s",";","d_fake_s",";",
                                  "gan_source_gen",";","gan_source_disc",";","dice","\n"])
    
    #Training Begins
    for epoch in range(0, EPOCHS):
        
        steps_per_epoch = train_dataloader.__len__()
        print(steps_per_epoch)
        print('Epoch {}'.format(epoch))
        start = time.time()
  
        progbar = Progbar(steps_per_epoch)
        
        d_real = []
        d_fake = []
        gan_gen = []
        gan_disc = []
        dice = []
        
        for batch in range(steps_per_epoch):
  
              X_img, X_prior, [X_gt_img, Y_real], _ = next(train_gen)
             
              X_fake, Y_fake = generate_fake_samples(g_model, 
                                                     [X_img, X_prior], d_model.output_shape[1])
              
              # update discriminator for real samples
              d_loss_real = d_model.train_on_batch([X_img, X_gt_img], Y_real)
              d_real.append(d_loss_real[0])
              
              # update discriminator for generated samples
              d_loss_gen_source = d_model.train_on_batch([X_img, X_fake], Y_fake)
              d_fake.append(d_loss_gen_source[0])
              
              # update the generator
              gan_result = gan_model.train_on_batch([X_img, X_prior],
                                                    [np.ones_like(Y_real), X_gt_img])
              gan_gen.append(gan_result[2])
              gan_disc.append(gan_result[1])
              dice.append(gan_result[-1])
              
              # summarize performance              
              progbar.add(1, values=[("D_real", d_real[-1]), ("D_fake", d_fake[-1]),
                                     ("Gan_gen", gan_gen[-1]),("Gan_disc", gan_disc[-1]), 
                                     ("Dice", dice[-1])])
            
        d_real = np.array(d_real)
        d_fake = np.array(d_fake)
        gan_gen = np.array(gan_gen)
        gan_disc = np.array(gan_disc)
        dice = np.array(dice)
        
        with open(RESULTS_PATH + 'logtrain.txt', 'a') as log_train:
            log_train.writelines([str(epoch),";",str(d_real.mean()),";",str(d_fake.mean()),";",
                                  str(gan_gen.mean()),";",str(gan_disc.mean()),";",
                                  str(dice.mean()),"\n"])
  
        print('Epoch %s/%s, Time: %s' % (epoch , EPOCHS, time.time() - start))
        

       # save weights on every 2nd epoch and validate
        if epoch % 2 == 0:
            model_weights_path = os.path.join(WEIGHTS_PATH + 'gen_weights_epoch_%s.h5' % (epoch))
            g_model.save_weights(model_weights_path, overwrite=True)
            
            model_weights_path = os.path.join(WEIGHTS_PATH + 'disc_weights_epoch_%s.h5' % (epoch))
            d_model.save_weights(model_weights_path, overwrite=True)
            
            if(args.validation):
                
                steps_for_validation = val_dataloader.__len__()
                progbar = Progbar(steps_for_validation)
                val_loss_gen = []
                val_loss_disc = []
                val_dice = []
        
                for step in range(steps_for_validation):

                    X_val_img, X_val_prior, [X_val_gt_img, Y_val_real], _ = next(val_gen)
                    val_result = gan_model.evaluate(x =[X_val_img, X_val_prior], 
                                                    y = [Y_val_real, X_val_gt_img], verbose = 1)
                    val_loss_gen.append(val_result[1])
                    val_loss_disc.append(val_result[2])
                    val_dice.append(val_result[6])
                    progbar.add(1, values=[("disc_loss:", val_result[1]),
                                           ("gen_loss:", val_result[2]),("Dice: ", val_result[6])])
        
                val_loss_gen = np.array(val_loss_gen)
                val_loss_disc = np.array(val_loss_disc)
                val_dice = np.array(val_dice)
                print("Validation:\n Dice:" + str(val_dice.mean()) + "\n")
        
                with open(RESULTS_PATH + 'logval.txt', 'a') as log_val:
                    log_val.writelines([str(epoch),';', str(val_loss_disc.mean()),';', 
                                        str(val_loss_gen.mean()),';', str(val_dice.mean()),'\n'])
    
    return 0

def train_contour_propagation(WEIGHTS_PATH, RESULTS_PATH, args):
    
    ### Define Parameters
    
    EPOCHS = args.epochs
    INPUT_DIM = [args.crop_height, args.crop_width, 2]
    lr = args.lrcpp
    func_dict = {
            'dice_loss' : dice_loss,
            'mae' : 'mae',
            'vgg_loss' : vgg_loss
            }
    
    loss = func_dict[args.segmentation_loss]
    
    #Define Models
    m = unet(image_shape = INPUT_DIM, activation_fn = 'sigmoid')
    m.compile(optimizer = Adam(lr = lr), loss = loss, metrics = [dice_coef])

    if(args.load_pretrained):
        m.load_weights(args.pretrained_cpp_weight_path) 
    
    #Get data generators
    if(args.validation):
        train_dataloader, val_dataloader = get_data(args, contprop = True)
    else: 
        train_dataloader = get_data(args, contprop = True)

    #Defining Callbacks
    weights_path = WEIGHTS_PATH + 'Chk.h5'
    checkpoint = ModelCheckpoint(weights_path,  
                                 monitor='val_loss', 
                                 verbose=1, save_best_only=True)
    csv_logger = CSVLogger(RESULTS_PATH + 'log.out', 
                           append=True, separator=';')
    #earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,
    #                              min_delta = 0.0001, patience = 10, mode = 'min')
    callbacks_list = [checkpoint, csv_logger]
    print(train_dataloader.__len__())
    result = m.fit(train_dataloader, epochs=EPOCHS, steps_per_epoch = train_dataloader.__len__(),
                              validation_data = val_dataloader, callbacks=callbacks_list, 
                              verbose = True)

    model_name = os.path.join(WEIGHTS_PATH + 'cpp_model.h5')
    m.save(model_name)
    
    return 0
    
    

def main():
    
    args = get_args()
    model = args.model
    
    save_path = args.save_path_top + args.save_path_add + '/'
    
    if(os.path.exists(save_path)):
        WEIGHTS_PATH = os.path.join(save_path, 'weights/')
        RESULTS_PATH = os.path.join(save_path, 'results/')
        if(not os.path.exists(WEIGHTS_PATH)):
            os.mkdir(WEIGHTS_PATH)
        if(not os.path.exists(RESULTS_PATH)):
            os.mkdir(RESULTS_PATH)
    else:
        os.mkdir(save_path)
        WEIGHTS_PATH = os.path.join(save_path, 'weights/')
        RESULTS_PATH = os.path.join(save_path, 'results/')
        if(not os.path.exists(WEIGHTS_PATH)):
            os.mkdir(WEIGHTS_PATH)
        if(not os.path.exists(RESULTS_PATH)):
            os.mkdir(RESULTS_PATH)
    
    if(model == 'ContourPropagation'):
        train_contour_propagation(WEIGHTS_PATH, RESULTS_PATH, args)
    elif(model == 'GuidedPix2Pix'):
        train_guided_pix2pix(WEIGHTS_PATH, RESULTS_PATH, args)
    else:
        print("Specify valid model name: ContourPropagation or GuidedPix2Pix ")
    
    
if __name__ == '__main__':
    main()       
    

