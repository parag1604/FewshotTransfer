import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import argparse
import matplotlib.pyplot as plt
import os, sys

# Reusing componets from other files
from utils import *
from model import *
# from train_unet import validate, test
# from train_monet import validate_monet, test_monet

file = open('temp.pkl', 'wb')
pickle.dump([], file)
file.close()

# Parsing cmd line arguments
parser = argparse.ArgumentParser(usage=\
'''python evaluate.py
        --model-type=<unet/monet>
        --test-data-path=<path_to_dir>
        --model-path=<path_to_dir>''',
        description='Evaluate a trained model')
parser.add_argument('--model-type',
                    type=str,
                    help='evalaute unet or monet model')
parser.add_argument('--test-data-path',
                    type=str,
                    help='the path to validation slices root directory\
                     (subfolders must include imgs, seg_l, seg_w)')
parser.add_argument('--model-path',
                    type=str,
                    help='the path to trained model to be evaluated')
args=parser.parse_args()

load_path = args.model_path
model_type = args.model_type
test_path = args.test_data_path

ngpu = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(ngpu, "GPU available" if ngpu == 1 else "GPUs available")
print("Using device -", device)

batch_size = 5

# One step of unet validation
def validate(validate_path = 'data/validate/', pre=False, test=True):
    ''' Validation step'''
    mode = 'l' if pre else 'w'
    model.eval()
    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0
    val_avg_effort_ratio = 0.0
    val_avg_effort_pixels = 0.0
    if test:
        print('------------ TESTING --------------')
    else:
        print('---------- VALIDATING ------------')
    for j, (imgs, segs) in enumerate(load_batch(device,
                                                batch_size,
                                                data_path=validate_path,
                                                mode=mode)):
        outputs = model(imgs)
        loss = bce_dice_loss(segs, outputs)
        dscoeff = dice_coeff(segs, outputs, smooth=0).item()
        effort_ratio, effort_pixels = annotation_effort(segs, outputs)
        if test:
            print("Iter:", j+1, "| loss:", round(loss.item(), 4),\
                  "| dsc:", round(dscoeff, 4))
        else:
            print("Epoch:", epoch, "| Iter:", j+1, "| loss:", \
                  round(loss.item(), 4), "| dsc:", round(dscoeff, 4))
        val_avg_loss += loss.item()
        val_avg_dscoeff += dscoeff
        val_avg_effort_ratio += effort_ratio
        val_avg_effort_pixels += effort_pixels
    val_avg_loss = round(val_avg_loss/(j+1), 4)
    val_avg_dscoeff = round(val_avg_dscoeff/(j+1), 4)
    val_avg_effort_ratio = round(val_avg_effort_ratio/(j+1), 4)
    val_avg_effort_pixels = round(65536*val_avg_effort_pixels/(j+1), 4)
    print('-------------- DONE --------------')
    return val_avg_loss, val_avg_dscoeff,\
            val_avg_effort_ratio, val_avg_effort_pixels

# Reusing unet validation for testing
def test(test_path = 'data/test/', pre=False, test=True):
    '''Reusing validation function for testing'''
    print()
    test_avg_loss, test_avg_dscoeff, test_avg_effort_ratio,\
    test_avg_effort_pixels = validate(test_path, pre=pre, test=test)
    print("Test_loss:", test_avg_loss, "| Test_dsc:", test_avg_dscoeff)
    print("Average Correction Effort Ratio:", test_avg_effort_ratio)
    print("Average Correction Effort in pixels per slice:",\
            test_avg_effort_pixels, "out of 65536 pixels")


# One step of monet validation
def validate_monet(validate_path = 'data/validate/', pre=False, test=True):
    Lambda = 0.8
    mode = 'l' if pre else 'w'
    if pre:
        model.eval()
    else:
        model_enc.eval()
        model_dec1.eval()
        model_dec2.eval()
    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0
    if test:
        print('------------ TESTING --------------')
    else:
        print('---------- VALIDATING ------------')
    for j, (imgs, p_segs, segs) in enumerate(load_batch(device,
                                                batch_size,
                                                data_path=validate_path,
                                                monet=True,
                                                mode=mode)):
        if pre:
            outputs = model(imgs)
            loss = bce_dice_loss(p_segs, outputs)
            dscoeff = dice_coeff(p_segs, outputs).item()
        else:
            outputs1 = model_dec1(model_enc(imgs))
            outputs2 = model_dec2(model_enc(imgs))
            loss1 = bce_dice_loss(p_segs, outputs1)
            loss2 = bce_dice_loss(segs, outputs2)
            loss = ((1 - Lambda) * loss1) + (Lambda * loss2)
            dscoeff = dice_coeff(segs, outputs2, smooth=0).item()
        if test:
            print("Iter:", j+1, "| loss:", round(loss.item(), 4),\
                  "| dsc:", round(dscoeff, 4))
        else:
            print("Epoch:", epoch, "| Iter:", j+1, "| loss:", \
                  round(loss.item(), 4), "| dsc:", round(dscoeff, 4))
        val_avg_loss += loss.item()
        val_avg_dscoeff += dscoeff
    val_avg_loss = round(val_avg_loss/(j+1), 4)
    val_avg_dscoeff = round(val_avg_dscoeff/(j+1), 4)
    print('-------------- DONE --------------')
    return val_avg_loss, val_avg_dscoeff

# Reusing monet validation for testing
def test_monet(test_path = 'data/test/', pre=False, test=True):
    print()
    test_avg_loss, test_avg_dscoeff = validate_monet(test_path,\
                                                     pre=pre, test=test)
    print("Test_loss:", test_avg_loss, "| Test_dsc:", test_avg_dscoeff)

# Loading model and parameters
if model_type == "unet": # vanilla unet
    model = UNet2D(1).to(device)
    model.load_state_dict(torch.load(load_path))
    test(test_path, pre=False) # testing unet
elif model_type == "monet": # monet
    model_enc = MO_Net_encoder().to(device)
    model_dec2 = MO_Net_decoder(1).to(device)
    model_enc.state_dict(torch.load(load_path.replace('.', '_enc.')))
    model_dec2.state_dict(torch.load(load_path.replace('.', '_dec.')))
    test_monet(test_path, pre=False, test=True) # testing monet
else: # exception handling
    print("Invalid option for model")
    print("Valid options - <unet>, <monet>")

os.system("rm temp.pkl")
