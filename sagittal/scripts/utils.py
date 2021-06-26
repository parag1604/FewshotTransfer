import torch
import torch.nn.functional as F

import numpy as np
from PIL import Image, ImageOps
import os
import pickle

def load_batch(device, batch_size = 3, dims=(256, 256),\
               data_path='data/train/', mode="w", monet=False,\
               sup=1.0, method="vol-s", train=False):
    '''
    - mode(s of operation):
        - pre-training: "l"(umbar spine)
        - fine-tining: "w"(hole spine)
    - supervision: (0,1] (fraction of supervision allowed)
    - method (fine-tuning sample selection):
        - "vol-r" (random volumes)
        - "vol-s" (sequential volumes)
        - "slice-r" (random slices per volume)
        - "slice-c" (central slices of every volume)
    '''
    file = open('temp.pkl', 'rb')
    img_names = pickle.load(file)
    file.close()
    if len(img_names) > 0 and train: # files have already been selected
        img_names = np.array(img_names)
    # Two cases of partial supervision
    #   Pretraining supervision (volume wise)
    #   Fine tuning supervision (2 types)
    #       Volume wise supervision (2 subtypes)
    #           1. Sequential volumes
    #           2. Random volumes
    #       Slice wise supervision (2 subtypes)
    #           3. Random slices
    #           4. Central slices
    elif train == True and sup < 1.0: # Ensure partially supervised  Training mode
        if mode == 'l': # pre-training
            img_names = sorted(os.listdir(os.path.join(data_path, 'img')))
            img_names = img_names[:int(len(img_names)*sup)]
        else: # fine-tuning
            if method == "vol-s":
                img_names = sorted(os.listdir(os.path.join(data_path, 'img')))
                img_names = img_names[:int(len(img_names)*sup)]
            else:
                slices_per_volume = 100 # considered 100 slices per vol
                all_slices = sorted(os.listdir(os.path.join(data_path, 'img')))
                num_volumes = len(all_slices) // slices_per_volume
                img_names = []
                if method == "vol-r":
                    rnd_perm = np.random.permutation(num_volumes)
                    for i in rnd_perm[:int(num_volumes * sup)]:
                        img_names.extend(all_slices[\
                            slices_per_volume * i : slices_per_volume * (i + 1)
                        ])
                elif method == "slice-r":
                    for i in range(num_volumes):
                        rnd_perm = np.random.permutation(slices_per_volume)
                        slices = rnd_perm[:int(slices_per_volume * sup)]
                        for slice in slices:
                            img_names.append(
                                all_slices[i * slices_per_volume + slice])
                elif method == "slice-c":
                    vol_idxs = np.array(list(range(slices_per_volume)))
                    semi_sup = int(100*sup) // 2
                    middle_idx = slices_per_volume // 2
                    starting_idx = middle_idx - semi_sup
                    ending_idx = middle_idx + semi_sup
                    slices = vol_idxs[starting_idx - 1 : ending_idx]
                    for i in range(num_volumes):
                        for slice in slices:
                            img_names.append(
                                all_slices[i * slices_per_volume + slice])
        file = open('temp.pkl', 'wb')
        pickle.dump(img_names, file)
        file.close()
        img_names = np.array(img_names)
    else: # for validation/testing, return the entire folder
        img_names = np.array(os.listdir(os.path.join(data_path, 'img')))
    mask_names = np.array([x for x in img_names])
    # Shuffel the batch for effective SGD
    shuffled_idxs = np.random.permutation(len(img_names))
    shuffled_img_names = img_names[shuffled_idxs]
    shuffled_mask_names = mask_names[shuffled_idxs]
    no_of_batches = img_names.shape[0] // batch_size
    extra_batch = img_names.shape[0] % batch_size
    for i in range(no_of_batches):
        imgs, segs, seg_ws, seg_ls = [], [], [], []
        for j in range(batch_size):
            im = Image.open(os.path.join(
                data_path, 'img', shuffled_img_names[i*batch_size+j]))
            if monet:
                seg_l = Image.open(os.path.join(data_path,\
                    'seg_l', shuffled_mask_names[i*batch_size+j]))
                seg_w = Image.open(os.path.join(data_path,\
                    'seg_w', shuffled_mask_names[i*batch_size+j]))
            else:
                seg = Image.open(os.path.join(data_path,\
                    'seg_'+mode, shuffled_mask_names[i*batch_size+j]))
            im = ImageOps.grayscale(im)
            im = im.resize(dims)
            im = np.array(im) / 255.
            imgs.append(im)
            if monet:
                seg_l = seg_l.resize(dims)
                seg_l = np.array(seg_l) / 255.
                seg_ls.append(seg_l)
                seg_w = seg_w.resize(dims)
                seg_w = np.array(seg_w) / 255.
                seg_ws.append(seg_w)
            else:
                seg = seg.resize(dims)
                seg = np.array(seg) / 255.
                segs.append(seg)
        if monet:
            yield (torch.FloatTensor(imgs).view(batch_size,
                                                1,dims[0],dims[1]).to(device),
                   torch.FloatTensor(seg_ls).view(batch_size,
                                                1,dims[0],dims[1]).to(device),
                   torch.FloatTensor(seg_ws).view(batch_size,
                                                1,dims[0],dims[1]).to(device))
        else:
            yield (torch.FloatTensor(imgs).view(batch_size,
                                                1,dims[0],dims[1]).to(device),
                   torch.FloatTensor(segs).view(batch_size,
                                                1,dims[0],dims[1]).to(device))
    if extra_batch:
        imgs, segs, seg_ws, seg_ls = [], [], [], []
        for j in range(extra_batch):
            im = Image.open(os.path.join(
                data_path, 'img', shuffled_img_names[(i+1)*batch_size]))
            if monet:
                seg_l = Image.open(os.path.join(data_path,\
                    'seg_l', shuffled_mask_names[i*batch_size+j]))
                seg_w = Image.open(os.path.join(data_path,\
                    'seg_w', shuffled_mask_names[i*batch_size+j]))
            else:
                seg = Image.open(os.path.join(data_path,\
                    'seg_'+mode, shuffled_mask_names[i*batch_size+j]))
            im = ImageOps.grayscale(im)
            im = im.resize(dims)
            im = np.array(im) / 255.
            imgs.append(im)
            if monet:
                seg_l = seg_l.resize(dims)
                seg_l = np.array(seg_l) / 255.
                seg_ls.append(seg_l)
                seg_w = seg_w.resize(dims)
                seg_w = np.array(seg_w) / 255.
                seg_ws.append(seg_w)
            else:
                seg = seg.resize(dims)
                seg = np.array(seg) / 255.
                segs.append(seg)
        if monet:
            yield (torch.FloatTensor(imgs).view(extra_batch,
                                                1,dims[0],dims[1]).to(device),
                   torch.FloatTensor(seg_ls).view(extra_batch,
                                                1,dims[0],dims[1]).to(device),
                   torch.FloatTensor(seg_ws).view(extra_batch,
                                                1,dims[0],dims[1]).to(device))
        else:
            yield (torch.FloatTensor(imgs).view(extra_batch,
                                                1,dims[0],dims[1]).to(device),
                   torch.FloatTensor(segs).view(extra_batch,
                                                1,dims[0],dims[1]).to(device))


def dice_coeff(y_true, y_pred, smooth = 1.):
    '''
    Function to calculate modified dice similarity coefficient
    The smooth value is Laplacian Smoothness (generally used for naive bayes)
    '''
    assert y_true.shape == y_pred.shape, "Tensor dimensions must match"
    y_pred = y_pred > 0.5
    shape = y_true.shape
    y_true_flat = y_true.view(shape[0]*shape[1]*shape[2]*shape[3],)
    y_pred_flat = y_pred.view(shape[0]*shape[1]*shape[2]*shape[3],)
    intersection = torch.sum(y_true_flat * y_pred_flat)
    score = (2. * intersection + smooth) / (torch.sum(y_true_flat) +\
                                            torch.sum(y_pred_flat) + 2 * smooth)
    return score

def dice_loss(y_true, y_pred):
    '''
    Smooth dice loss
    '''
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    '''
    Custom loss (dice loss + binary cross entropy) (for all pixel in a slice)
    '''
    loss = F.binary_cross_entropy(y_pred, y_true) + dice_loss(y_true, y_pred)
    return loss

def annotation_effort(y_true, y_pred):
    '''
    Function to calculate relative and pixel-wise annotation effort
    '''
    assert y_true.shape == y_pred.shape, "Tensor dimensions must match"
    y_pred = y_pred > 0.5
    shape = y_true.shape
    y_true_flat = y_true.view(shape[0]*shape[1]*shape[2]*shape[3],)
    y_pred_flat = y_pred.view(shape[0]*shape[1]*shape[2]*shape[3],)
    actual_pos = torch.sum(y_true_flat==1.)
    true_pos = torch.sum(y_true_flat[y_pred_flat==True]==1.)
    true_neg = torch.sum(y_true_flat[y_pred_flat==False]==0.)
    false_pos = torch.sum(y_true_flat[y_pred_flat==True]==0.)
    false_neg = torch.sum(y_true_flat[y_pred_flat==False]==1.)
    mistakes = false_pos.item() + false_neg.item()
    total_pixels = true_pos.item() + false_neg.item() + false_pos.item() +\
                    true_neg.item()
    return mistakes/actual_pos.item(), mistakes/total_pixels
