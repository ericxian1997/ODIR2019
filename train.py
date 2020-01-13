# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import dataset_processing
from torch.utils.data import DataLoader
from model import ft_net, ft_net_152, ft_efnet
#from random_erasing import RandomErasing
import json
from shutil import copyfile

version =  torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='efnetb3_adam3', type=str, help='output model name')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
#parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
opt = parser.parse_args()


DATA_PATH = '/home/yuqiao/Code/odir_code/data'
TRAIN_DATA = 'train_img'
TEST_DATA = 'test_img'
TRAIN_IMG_FILE = 'train_img.txt'
TEST_IMG_FILE = 'test_img.txt'
TRAIN_LABEL_FILE = 'train_label.txt'
TEST_LABEL_FILE = 'test_label.txt'


use_gpu = torch.cuda.is_available()

seed = 1997
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
#print(gpu_ids[0])


######################################################################
# Load Data
# ---------
#

transformations = [transforms.Resize((512,512), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


transformations=  transforms.Compose(transformations)

#if opt.erasing_p>0:
#    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list


image_datasets = {}

image_datasets['train'] = dataset_processing.DatasetProcessing(
    DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, transformations)

#image_datasets['val'] = dataset_processing.DatasetProcessing(
#    DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transformations)

train_loader = DataLoader(image_datasets['train'],
                          batch_size=opt.batchsize,
                          shuffle=True,
                          num_workers=4
                         )
'''
test_loader = DataLoader(dset_test,
                         batch_size=opt.batchsize,
                         shuffle=False,
                         num_workers=4
                         )
'''

dataset_sizes = len(image_datasets['train'])




since = time.time()
inputs_l, inputs_r, classes = next(iter(train_loader))
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in train_loader:
                # get the inputs
                inputs_l, inputs_r, labels = data
                now_batch_size,c,h,w = inputs_l.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs_l = Variable(inputs_l.cuda())
                    inputs_r = Variable(inputs_r.cuda())
                    labels = Variable(labels.cuda().float())
                else:
                    inputs_l, inputs_r, labels = Variable(inputs_l), Variable(inputs_r), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs_l, inputs_r)

                #class_weights = torch.tensor([8.77,8.66,46.0,47.0,60.0,97.0,57.0,10.0])
                #class_weights = class_weights / sum(class_weights)
                #class_weights = class_weights.cuda()
                #print(outputs)
                loss = criterion(outputs, labels).mean()
                #loss = (loss*class_weights).mean()
                
                #print(loss)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if int(version[2]) > 3: # for the new version like 0.4.0 and 0.5.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size


            epoch_loss = running_loss / dataset_sizes
            
            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss))
            
            y_loss[phase].append(epoch_loss)          
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
    fig.savefig( os.path.join('./model',name,'train.jpg'))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

model = ft_efnet(8)


print(model)

if use_gpu:
    model = model.cuda()

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss(reduction='none')


ignored_params = list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}])


# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./model',name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
#record every run
copyfile('./train.py', dir_name+'/train.py')
copyfile('./model.py', dir_name+'/model.py')

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=40)

