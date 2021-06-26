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
from argparse import ArgumentParser
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

def get_args():
    parser = ArgumentParser(description = 'GPix2Pix_ArgumentParser')
    parser.add_argument('--save_path_add', type=str)
    parser.add_argument('--test_data_path', type=str, default = './Resampled_Vols/')
    parser.add_argument('--model', type=str)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--image_guide', type=bool, default=False)
    parser.add_argument('--mask_guide', type=bool, default=False)
    parser.add_argument('--q', type=int, default=3)
    parser.add_argument('--weights_path', type=str)
    parser.add_argument('--test_baseline', type=bool, default = False)

    args = parser.parse_args()
    return args

def main():
    
    #Allow GPU Growth
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)
    
    args = get_args()
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
        effort = np.ones(imgvol.shape[2])
        falsep = np.zeros(imgvol.shape[2])
        falsen = np.zeros(imgvol.shape[2])
        totalp = np.zeros(imgvol.shape[2])
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
            fp, fn, total, e = inpainting_effort(gen, seg, 0.5)
            effort[i] = e
            totalp[i] = total
            falsep[i] = fp
            falsen[i] = fn
            
        #save(ivol, save_path + str(num) + 'image.nii.gz')
        #save(mvol, save_path + str(num) + 'mask.nii.gz')
        #save(predvolg, save_path + str(num) + 'predvol.nii.gz')
        
        dice_scores[index] = diceg.mean()
        correction_scores[index] = (np.sum(falsep) + np.sum(falsen))/np.sum(totalp)
        end = time.time()
        print(fname, dice_scores[index], correction_scores[index])
        print("Execution Time:" , end- start)
        
        
        with open(save_path + 'Result.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([fname])
            writer.writerow(diceg)
            writer.writerow(["DSC-test-mean and var"])
            writer.writerow([diceg.mean(), diceg.var()])
            writer.writerow(["Inpainting_Effort"])
            writer.writerow(effort)
            writer.writerow(["Inpainting-mean and var"])
            writer.writerow([effort.mean(), effort.var()])
            writer.writerow(["False Positives"])
            writer.writerow(falsep)
            writer.writerow(["False Positives-mean and sum"])
            writer.writerow([falsep.mean(), np.sum(falsep)])
            writer.writerow(["False Negatives"])
            writer.writerow(falsen)
            writer.writerow(["False Negatives-mean and sum"])
            writer.writerow([falsen.mean(), np.sum(falsen)])
            writer.writerow(["Total", np.sum(totalp)])
            writer.writerow(["Inpainting", (np.sum(falsep) + np.sum(falsen))/np.sum(totalp)])

    with open(save_path + 'Result.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(dice_scores)
        writer.writerow([dice_scores.mean()])
        writer.writerow(correction_scores)
        writer.writerow([correction_scores.mean()])
        
if __name__ == '__main__':
    print("main")
    main()  