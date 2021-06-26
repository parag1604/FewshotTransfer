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
'''python train_unet.py <epochs> <batch_size>
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

# Creating model and optimizer
model = UNet2D(1).to(device)
if not is_pre_training:
    model.load_state_dict(torch.load(load_path))
optimizer = optim.Adam(model.parameters(), learning_rate)

# choice of files should be consistent across all epochs
file = open('temp.pkl', 'wb')
pickle.dump([], file)
file.close()

def train(train_path = 'data/train/', pre=False):
    '''Training step'''
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

def validate(validate_path = 'data/validate/', pre=False, test=False):
    ''' Validation step'''
    mode = 'l' if pre else 'w'
    model.eval()
    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0
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
        dscoeff = dice_coeff(segs, outputs).item()
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

def test(test_path = 'data/test/', pre=False, test=True):
    '''Reusing validation function for testing'''
    print()
    test_avg_loss, test_avg_dscoeff = validate(test_path, pre=pre, test=test)
    print("Test_loss:", test_avg_loss, "| Test_dsc:", test_avg_dscoeff)

losses = [] # Model training history storage variables
dscoeffs = []
avg_losses = []
val_avg_losses = []
avg_dscoeffs = []
val_avg_dscoeffs = []

best_dscoeff = 0.0 # Vanilla training loop with best model retaining capability
for epoch in range(1, epochs+1):
    print('---------- EPOCH:', epoch, '----------')
    avg_loss, avg_dscoeff = train(train_path, pre=is_pre_training)
    val_avg_loss, val_avg_dscoeff = validate(validate_path, pre=is_pre_training)
    avg_losses.append(avg_loss)
    val_avg_losses.append(val_avg_loss)
    avg_dscoeffs.append(avg_dscoeff)
    val_avg_dscoeffs.append(val_avg_dscoeff)
    print("Epoch:", epoch, "| loss:", avg_loss, "| dsc:", avg_dscoeff,
          "| val_loss:", val_avg_loss, "| val_dsc:", val_avg_dscoeff)
    print()
    if best_dscoeff < val_avg_dscoeff:
    	best_dscoeff = val_avg_dscoeff
    	torch.save(model.state_dict(), output_path)

print("Model saved with validation dice coeff =", best_dscoeff)
print()

model.load_state_dict(torch.load(output_path)) # Loading best model and testing
test(test_path, pre=is_pre_training, test=True)

plt.plot(avg_dscoeffs)    # Plotting graphs
plt.plot(val_avg_dscoeffs)
plt.title('DSC plot during training')
plt.xlabel('Epochs')
plt.ylabel('dice similarity coefficient')
plt.legend(['training', 'testing'])
plt.savefig('dsc.png')
plt.show()

os.system("rm temp.pkl") # Removing temp files
