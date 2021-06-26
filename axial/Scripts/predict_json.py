# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 08:17:08 2020

@author: Mehak

Using every (2*q) th slice as supervision, predicts segmentations for the entire test set.

"""

from models import *
import os
import csv
from medpy.io import load, save
from albumentations import (Compose, PadIfNeeded)
from utils import *
import json
import numpy as np
import time

def preprocess(img, lower = -500, upper= 1000):
    img[img < lower] = lower
    img[img > upper] = upper
    img = (img)/upper
    return img


def resize(input_image, height, width):
    resized = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return resized

#Arguments
class Dict2Class(object):
    
    def __init__(self, args_dict):
        
        for key in args_dict:
            setattr(self, key, args_dict[key])
        
def read_json():
    
    path = './args_test.json'
    f = open(path, )
    args_dict = json.load(f)
    args = Dict2Class(args_dict)
    
    return args

def main():
    
    #Allow GPU Growth
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)
    
    args = read_json()
    h = args.height
    w = args.width
    q = args.q
    model = args.model
    
    if(model != 'ContourPropagation' and model != 'GuidedPix2Pix'):
        print("Invalid method. Please specify 'ContourPropagation' or 'GuidedPix2Pix'")
        return 0
    
    save_path = './Out_predict/' + args.save_path_add + '/'
    if(not os.path.exists(save_path)):
        os.mkdir(save_path)

    test_data_img = args.test_data_path + 'Image/'
    test_data_mask = args.test_data_path + 'Mask/'
    
    test_cases = os.listdir(test_data_img)
    dice_scores = np.zeros(len(test_cases))
    correction_scores = np.zeros(len(test_cases))
    sensitivity = np.zeros(len(test_cases))
    specificity = np.zeros(len(test_cases))
    accuracy = np.zeros(len(test_cases))
    
    if(model == 'ContourPropagation'):
        gmodel = unet(image_shape = (h,w,2), activation_fn = 'sigmoid')
        gmodel.load_weights(args.weights_path)
    elif(model == 'GuidedPix2Pix'):
        gmodel = unet_withBFT(image_shape = (h,w,1))
        gmodel.load_weights(args.weights_path)
    
    index = -1
    for fname in test_cases:
        
        index = index + 1
        start = time.time()
        imgvol, target_header = load(test_data_img +  fname)
        wholemask, whole_header = load(test_data_mask + fname)
        imgvol = preprocess(imgvol)
        predvolg = np.zeros((h,w, imgvol.shape[2]))    
        ivol = np.zeros((h,w, imgvol.shape[2]))
        mvol = np.zeros((h,w, imgvol.shape[2]))
        seg = np.zeros((1,h,w,1))
        
        diceg = np.zeros(imgvol.shape[2])
        true_positives = np.ones(imgvol.shape[2])
        true_negatives = np.zeros(imgvol.shape[2])
        false_positives = np.zeros(imgvol.shape[2])
        false_negatives = np.zeros(imgvol.shape[2])
        gt_slice = 10
        
        #starting from slice 10
        for i in range(11,imgvol.shape[2]):
            real_A_raw = imgvol[:,:,i]
            real_B_raw = np.float32(wholemask[:,:,i] > 0)
            
            mod = i % (2*q)
            
            if(mod == 0):
                gt_slice = i
                prior_raw = np.int32(wholemask[:,:,gt_slice] > 0)
            elif(mod <= q):
                prior_raw = np.int32(wholemask[:,:,gt_slice] > 0)
            else:
                if(gt_slice + 2*q < imgvol.shape[2]):
                    prior_raw = np.int32(wholemask[:,:,gt_slice + 2*q] > 0)
                else:
                    prior_raw = np.int32(wholemask[:,:,gt_slice] > 0)
                
            stack = tf.stack([real_A_raw, real_B_raw, prior_raw], axis = -1)
            r = resize(stack, h,w)
            
            ivol[:,:,i] = r[:,:,0]
            mvol[:,:,i] = r[:,:,1]
            
            if(not args.test_baseline):
                if(model == 'ContourPropagation'):
                    inp = np.empty((1,h,w,2))
                    inp[0,:,:,0] = r[:,:,0]
                    inp[0,:,:,1] = r[:,:,2]
                    seg[0,:,:,0] = r[:,:,1]
                    if(gt_slice == i):
                        gen = inp[:,:,:,1:]
                    else:
                        gen = gmodel.predict(inp)

                elif(model == 'GuidedPix2Pix'):
                    real_A = np.empty((1,h,w,1))
                    real_prior = np.empty((1,h,w,1))
                    real_A[0,:,:,0] = r[:,:,0]
                    seg[0,:,:,0] = r[:,:,1]
                    real_prior[0,:,:,0] = r[:,:,2]                    
                    if(gt_slice == i):
                        gen = real_prior
                    else:
                        if(args.mask_guide or args.image_guide == False):
                            gen = gmodel.predict([real_A, real_prior])
                        elif(args.image_guide):
                            gen = gmodel.predict([real_prior, real_A])
            else:
                gen = np.empty(1,h,w,1)
                gen[0,:,:,0] = r[:,:,2]
            
            predvolg[:,:,i] = gen[0,:,:,0]
            
            diceg[i] = dice_coef(gen, seg)
            fp_, fn_, tp_, tn_ = evaluation_metric(gen, seg, 0.5)
            false_positives[i] = fp_
            false_negatives[i] = fn_
            true_positives[i] = tp_
            true_negatives[i] = tn_
            
            
        #save(ivol, save_path + str(num) + 'image.nii.gz')
        #save(mvol, save_path + str(num) + 'mask.nii.gz')
        #save(predvolg, save_path + str(num) + 'predvol.nii.gz')
        
        eps = 0.001
        dice_scores[index] = diceg.mean()
        tp = np.sum(true_positives)
        tn = np.sum(true_negatives)
        fp = np.sum(false_positives)
        fn = np.sum(false_negatives)
        sensitivity[index] = (tp + eps) / (tp + fn + eps)
        specificity[index] = (tn + eps) / (tn + fp + eps)
        accuracy[index] = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        correction_scores[index] = (fp + fn + eps) / (tp + fn + eps)

        end = time.time()
        print(fname, dice_scores[index], correction_scores[index], sensitivity[index], specificity[index], accuracy[index])
        print("Execution Time:" , end- start)
        print("Num_slices : ", imgvol.shape[2])
        
        
        with open(save_path + 'Result.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fname])
            writer.writerow(["Dice : ", dice_scores[index]])
            writer.writerow(["Correction Effort", correction_scores[index]])
            writer.writerow(["Sensitivity: ", sensitivity[index]])
            writer.writerow(["Specificity: ", specificity[index]])
            writer.writerow(["Accuracy: ", accuracy[index]])


    with open(save_path + 'Result.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dice : ", dice_scores.mean()])
        writer.writerow(["Correction Effort", correction_scores.mean()])
        writer.writerow(["Sensitivity: ", sensitivity.mean()])
        writer.writerow(["Specificity: ", specificity.mean()])
        writer.writerow(["Accuracy: ", accuracy.mean()])
        
if __name__ == '__main__':
    print("main")
    main()  