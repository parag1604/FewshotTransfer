# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:26:36 2020

@author: Mehak

This script is to resample volumes, pad, normalize the volumes and save them as individual slices. 

The original volumes were named as :
    
CT volume: [filename1].nii.gz
Corresponding segmentation mask volume: [filename1]_seg.nii.gz

The script would need to be modified around line 50 for any other naming convention.

The slices are names according to the convention:
    
[volume_number]_slice[slice number].nii.gz

Image and mask slices are named the identically and saved in different folders.

"""
import os
import numpy as np
from medpy.io import save
from medpy.io import load
import medpy.filter
import cv2
from albumentations import (Compose, PadIfNeeded)

IMG_VOL_PATH = './Data/Original_Vols/Image/'
MASK_VOL_PATH = './Data/Original_Vols/Mask/'

resampled_imgvol = './Data/Resampled_Vols/Image/'
resampled_maskvol = './Data/Resampled_Vols/Mask/'  

#Only slices containing lumbar vertebra
lumbarimgpath = './Data/Axial_Slices/Lumbar/Image/'
lumbarmaskpath = './Data/Axial_Slices/Lumbar/Mask/'

#All slices of the volume
wholeimgpath = './Data/Axial_Slices/Whole/Image/'
wholemaskpath = './Data/Axial_Slices/Whole/Mask/'

#All slices except lumbar slices
restimgpath = './Data/Axial_Slices/Rest/Image/'
restmaskpath = './Data/Axial_Slices/Rest/Mask/'

with open('./newsliceslog.txt', 'a') as log_train:
    log_train.writelines(["Volume: ", ";", "dim1", ";", "dim2", ";","Afterpaddingdim", ";", "lumbar_start", ";", "lumbar_end", ";", "whole_start",";","whole_end",";","save_start",";","save_end", "\n"])

original_filenames = os.listdir(IMG_VOL_PATH)
original_segfilenames = os.listdir(MASK_VOL_PATH)
num = 0

for file in original_filenames:
    
    num = num + 1
    fname = file.split('.nii.gz')[0]
    imgvol_orig, imgvol_orig_header = load(IMG_VOL_PATH + fname + '.nii.gz')
    seg_orig, seg_orig_header = load(MASK_VOL_PATH + fname + '_seg.nii.gz')
    
    imgvol, imgvol_header = medpy.filter.image.resample(imgvol_orig, imgvol_orig_header, [1,1,1], 3)
    maskvol, maskvol_header = medpy.filter.image.resample(seg_orig, seg_orig_header, [1,1,1], 0)
    
    save(imgvol, resampled_imgvol + str(num) + '_vol.nii.gz', imgvol_header)
    save(maskvol, resampled_maskvol + str(num) + '_vol.nii.gz', maskvol_header)
    
    print(imgvol.shape)
    
    #Getting the vertebra number of each slice 
    labels = []
    #Lumbar vertebra labelled 20 to 24
    for i in range(maskvol.shape[2]):
        l = np.unique(maskvol[:,:,i])
        if 19 in l and 20 in l:
            labels.append(20)
        elif 25 in l and 24 in l:
            labels.append(24)
        else:
            labels.append(l[-1])
    
            
    lumbar_slices = np.copy(labels)
    
    #Slices with no anatomy
    zero_slices = np.copy(labels)
    zero_slices[zero_slices > 0] = 1
    zero_slices = 1-zero_slices
    
    #Slices with on lumbar vertebrae
    lumbar_slices[lumbar_slices < 20] = 0
    lumbar_slices[lumbar_slices > 24] = 0
    lumbar_slices[lumbar_slices > 0 ] = 1
    lumbar_indices = np.where(lumbar_slices == 1)[0]

    #To remove a few (10) slices in the beginning and end that dont have any anatamy
    diff =np.diff(zero_slices)
    
    if(-1 not in diff):
        start = 0
    else:
        start = np.where(diff == -1)[0][0]
    
    if(1 not in diff):
        end = maskvol.shape[2]-1
    else:
        end = np.where(diff == 1)[0][0] 
    
    if(start - 9 > 0):
        start_saving = start - 9
    else:
        start_saving = 0
    
    if(end + 10 < maskvol.shape[2]):
        end_saving = end + 10
    else:
        end_saving = maskvol.shape[2]-1
    
    #Pad to preserve aspect ratio before resizing
    dim1 = imgvol.shape[0]
    dim2 = imgvol.shape[1]
    
    dim = dim1 if dim1 > dim2 else dim2
    pad = Compose([PadIfNeeded(min_height = dim, min_width = dim, border_mode = cv2.BORDER_CONSTANT, value =-0.5, mask_value = 0)])
    with open('./newsliceslog.txt', 'a') as log_train:
        log_train.writelines([str(num), ";", str(dim1), ";", str(dim2), ";",str(dim), ";", str(lumbar_indices[0]), ";", str(lumbar_indices[-1]), ";", str(start),";",str(end),";",str(start_saving),";",str(end_saving), "\n"])
    
    chk = 0
    
    for i in range(start_saving, end_saving+1):
        
        k = i - start_saving
        img = imgvol[:,:,i]
        #Approximate HU of air and bone respectively
        img[img < -500] = -500
        img[img > 1000] = 1000
        img = img/1000
        padded = pad(image = img, mask = maskvol[:,:,i])
        
        if(i == lumbar_indices[0]):
            save(padded["image"], wholeimgpath + str(num) + '_slice' + str(k) + '.nii.gz')
            save(padded["mask"], wholemaskpath + str(num) + '_slice' + str(k) + '.nii.gz')
            save(padded["image"], lumbarimgpath + str(num) + '_slice' + str(chk) + '.nii.gz')
            save(padded["mask"], lumbarmaskpath + str(num) + '_slice' + str(chk) + '.nii.gz')
            chk = 1
        elif(i == lumbar_indices[-1]):
            save(padded["image"], wholeimgpath + str(num) + '_slice' + str(k) + '.nii.gz')
            save(padded["mask"], wholemaskpath + str(num) + '_slice' + str(k) + '.nii.gz')
            save(padded["image"], lumbarimgpath + str(num) + '_slice' + str(chk) + '.nii.gz')
            save(padded["mask"], lumbarmaskpath + str(num) + '_slice' + str(chk) + '.nii.gz')
            chk = 0
        else:
            if(chk == 0):
                save(padded["image"], wholeimgpath + str(num) + '_slice' + str(k) + '.nii.gz')
                save(padded["mask"], wholemaskpath + str(num) + '_slice' + str(k) + '.nii.gz')
                save(padded["image"], restimgpath + str(num) + '_slice' + str(k) + '.nii.gz')
                save(padded["mask"], restmaskpath + str(num) + '_slice' + str(k) + '.nii.gz')
            else:
                save(padded["image"], wholeimgpath + str(num) + '_slice' + str(k) + '.nii.gz')
                save(padded["mask"], wholemaskpath + str(num) + '_slice' + str(k) + '.nii.gz')
                save(padded["image"], lumbarimgpath + str(num) + '_slice' + str(chk) + '.nii.gz')
                save(padded["mask"], lumbarmaskpath + str(num) + '_slice' + str(chk) + '.nii.gz')
                chk = chk+1
    
    print(num, "done")
