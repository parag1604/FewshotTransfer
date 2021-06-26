import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import argparse
import matplotlib.pyplot as plt
import os

from utils import *
from model import *

# Getting command line arguments
parser = argparse.ArgumentParser(usage=\
'''python train_monet.py <epochs> <batch_size>
        --train-data-path=<path_to_dir>
        --val-data-path=<path_to_dir>
        --test-data-path=<path_to_dir>
        --output-path=<path_to_dir>
        [--pre-training]/[--model-path=<path_to_dir>]
        [--supervision=<fraction_of_supervision>]
        [--method=<method_of_fine_tuning_supervision>]''',
        description='Pre-train/fine-tune a UNet model')
parser.add_argument('epochs',
                    type=int,
                    help='number of epochs the model should train')
parser.add_argument('batch',
                    type=int,
                    help='training batch size per iteration')
parser.add_argument('--train-data-path',
                    type=str,
                    help='the path to training slices root directory\
                     (subfolders must include imgs, seg_l, seg_w)')
parser.add_argument('--val-data-path',
                    type=str,
                    help='the path to validation slices root directory\
                     (subfolders must include imgs, seg_l, seg_w)')
parser.add_argument('--test-data-path',
                    type=str,
                    help='the path to validation slices root directory\
                     (subfolders must include imgs, seg_l, seg_w)')
parser.add_argument('--output-path',
                    type=str,
                    help='the path to folder where model will be stored')
parser.add_argument('--pre-training',
                    action='store_true',
                    help='activate pre-training mode')
parser.add_argument('--model-path',
                    type=str,
                    help='the path to pre-trained model\
                     on which fine-tuning will be performed')
parser.add_argument('--method',
                    type=str,
                    help='''Fine tuning method, valid options:
vol-r (random volumes),
vol-s (sequential volumes),
slice-r (random slices per volume),
slice-c (central slices of every volume)''')
parser.add_argument("--supervision",
                    type=float,
                    help="amount of supervision to be used while training")

args = parser.parse_args()

is_pre_training = args.pre_training
if not is_pre_training:
    load_path = args.model_path
epochs = args.epochs
batch_size = args.batch
train_path = args.train_data_path
test_path = args.test_data_path
validate_path = args.val_data_path
output_path = args.output_path
train_method = args.method
supervision = args.supervision
learning_rate = 3e-4 if is_pre_training else 1e-4


ngpu = torch.cuda.device_count()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(ngpu, "GPU available" if ngpu == 1 else "GPUs available")
print("Using device -", device)

# Creating (compound) model and optimizers
model = UNet2D(1).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
if not is_pre_training: # Separate encoder and decoder for monet
    model.load_state_dict(torch.load(load_path))
    model_enc = MO_Net_encoder().to(device)
    model_dec1 = MO_Net_decoder(1).to(device)
    model_dec2 = MO_Net_decoder(1).to(device)
    # Extracting encoder and decoder parameters
    params1 = model.state_dict()
    params2 = model_enc.state_dict()
    params3 = model_dec1.state_dict()
    params4 = model_dec2.state_dict()
    for item in params1:
        if item in params2:
            params2[item] = params1[item]
    for item in params1:
        if item in params3:
            params3[item] = params1[item]
    for item in params1:       # 2nd decoder has same parameters
        if item in params4:    # as decoder 1, but a separate copy
            params4[item] = params1[item]
    model_enc.load_state_dict(params2)
    model_dec1.load_state_dict(params3)
    model_dec2.load_state_dict(params4)
    optimizer1 = optim.Adam(model_enc.parameters(), lr = learning_rate/10)
    optimizer2 = optim.Adam(model_dec1.parameters(), lr = learning_rate/3)
    optimizer3 = optim.Adam(model_dec2.parameters(), lr = learning_rate)

# choice of files should be consistent across all epochs
file = open('temp.pkl', 'wb')
pickle.dump([], file)
file.close()

# Same old one step of training
def pretrain_monet(train_path = 'data/train/', pre=True):
    mode = 'l' if pre else 'w'
    model.train()
    avg_loss = 0.0
    avg_dscoeff = 0.0
    print('----------- TRAINING -------------')
    for i, (imgs, segs) in enumerate(load_batch(device,
                                                batch_size,
                                                data_path=train_path,
                                                sup=supervision,
                                                mode=mode,
                                                train=True,
                                                method=train_method)):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = bce_dice_loss(segs, outputs)
        loss.backward()
        optimizer.step()
        dscoeff = dice_coeff(segs, outputs).item()
        print("Epoch:", epoch, "| Iter:", i+1, "| loss:", \
              round(loss.item(), 4), "| dsc:", round(dscoeff, 4))
        losses.append(loss.item())
        dscoeffs.append(dscoeff)
        avg_loss += loss.item()
        avg_dscoeff += dscoeff
    avg_loss = round(avg_loss/(i+1), 4)
    avg_dscoeff = round(avg_dscoeff/(i+1), 4)
    print('-------------- DONE --------------')
    model.eval()
    return avg_loss, avg_dscoeff

# Custom training for monet encoder decoder separately
def train_monet(train_path = 'data/train/', pre=False):
    mode = 'l' if pre else 'w'
    model_enc.train()
    model_dec1.train()
    model_dec2.train()
    avg_loss = 0.0
    avg_dscoeff = 0.0
    print('----------- TRAINING -------------')
    for i, (imgs, p_segs, segs) in enumerate(load_batch(device,
                                                batch_size,
                                                data_path=train_path,
                                                sup=supervision,
                                                mode=mode,
                                                monet=True,
                                                train=True,
                                                method=train_method)):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        outputs1 = model_dec1(model_enc(imgs))
        outputs2 = model_dec2(model_enc(imgs))
        loss1 = bce_dice_loss(p_segs, outputs1)
        loss2 = bce_dice_loss(segs, outputs2)
        loss = ((1 - Lambda) * loss1) + (Lambda * loss2)
        loss.backward()
        optimizer2.step()
        optimizer3.step()
        optimizer1.step()
        dscoeff = dice_coeff(segs, outputs2).item()
        print("Epoch:", epoch, "| Iter:", i+1, "| loss:", round(loss.item(), 4), "| dsc:", round(dscoeff, 4))
        losses.append(loss.item())
        dscoeffs.append(dscoeff)
        avg_loss += loss.item()
        avg_dscoeff += dscoeff
    avg_loss = round(avg_loss/(i+1), 4)
    avg_dscoeff = round(avg_dscoeff/(i+1), 4)
    print('-------------- DONE --------------')
    model_enc.eval()
    model_dec1.eval()
    model_dec2.eval()
    return avg_loss, avg_dscoeff

# One step of validation
def validate_monet(validate_path = 'data/validate/', pre=False, test=False):
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
            dscoeff = dice_coeff(segs, outputs2).item()
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

# Reusing validation for testing
def test_monet(test_path = 'data/test/', pre=False, test=True):
    print()
    test_avg_loss, test_avg_dscoeff = validate_monet(test_path,\
                                                     pre=pre, test=test)
    print("Test_loss:", test_avg_loss, "| Test_dsc:", test_avg_dscoeff)

# Model training history storage variable
losses = []
dscoeffs = []
avg_losses = []
val_avg_losses = []
avg_dscoeffs = []
val_avg_dscoeffs = []

# Vanilla training loop with best model retaining capability
best_dscoeff = 0.0
for epoch in range(1, epochs+1):
    print('---------- EPOCH:', epoch, '----------')
    if is_pre_training:
        avg_loss, avg_dscoeff = pretrain_monet(train_path, pre=is_pre_training)
        val_avg_loss, val_avg_dscoeff = validate_monet(validate_path, pre=is_pre_training)
    else:
        avg_loss, avg_dscoeff = train_monet(train_path, pre=is_pre_training)
        val_avg_loss, val_avg_dscoeff = validate_monet(validate_path, pre=is_pre_training)
    avg_losses.append(avg_loss)
    val_avg_losses.append(val_avg_loss)
    avg_dscoeffs.append(avg_dscoeff)
    val_avg_dscoeffs.append(val_avg_dscoeff)
    print("Epoch:", epoch, "| loss:", avg_loss, "| dsc:", avg_dscoeff,
          "| val_loss:", val_avg_loss, "| val_dsc:", val_avg_dscoeff)
    print()
    if best_dscoeff < val_avg_dscoeff:
    	best_dscoeff = val_avg_dscoeff
        if is_pre_training:
        	torch.save(model.state_dict(), output_path)
        else:
            torch.save(model_enc.state_dict(), output_path.replace('.', '_enc.'))
            torch.save(model_dec2.state_dict(), output_path.replace('.', '_dec.'))

print("Model saved with validation dice coeff =", best_dscoeff)
print()

# Loading best model and testing
if is_pre_training:
    model.load_state_dict(torch.load(output_path))
else:
    model_enc.load_state_dict(torch.load(output_path.replace('.', '_enc.')))
    model_dec2.load_state_dict(torch.load(output_path.replace('.', '_dec.')))
test_monet(test_path, pre=is_pre_training, test=True)

# Plotting graphs
plt.plot(avg_dscoeffs)
plt.plot(val_avg_dscoeffs)
plt.title('DSC plot during training')
plt.xlabel('Epochs')
plt.ylabel('dice similarity coefficient')
plt.legend(['training', 'testing'])
plt.savefig('dsc.png')
plt.show()

os.system("rm temp.pkl") # Removing temp files
