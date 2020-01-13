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
import time
import os
import scipy.io
import dataset_processing
from model import ft_net,ft_net_152, ft_efnet
from torch.utils.data import DataLoader
#from utils.OpCounter import OpCounter
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--name', default='efnetb3_adam2', type=str, help='save model path')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
opt = parser.parse_args()


DATA_PATH = 'data'
TRAIN_DATA = 'train_img'
TEST_DATA = 'test_img'
TRAIN_IMG_FILE = 'train_img.txt'
TEST_IMG_FILE = 'test_img.txt'
TRAIN_LABEL_FILE = 'train_label.txt'
TEST_LABEL_FILE = 'test_label.txt'

#which_epoch = opt.which_epoch
name = opt.name



######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#



transformations = [transforms.Resize((224,224), interpolation=3),
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
#
'''
image_datasets['train'] = dataset_processing.DatasetProcessing(
    DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, transformations)
'''
image_datasets['test'] = dataset_processing.DatasetProcessingTest(
   DATA_PATH, TEST_DATA, TEST_IMG_FILE, transformations)
'''
train_loader = DataLoader(image_datasets['train'],
                          batch_size=opt.batchsize,
                          shuffle=True,
                          num_workers=4
                         )
'''
test_loader = DataLoader(image_datasets['test'],
                         batch_size=opt.batchsize,
                         shuffle=False,
                         num_workers=4
                         )


dataset_sizes = len(image_datasets['test'])



use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    #network.load_state_dict(torch.load(save_path))

    # multi-GPU
    network = LoadDict(network, torch.load(save_path))
    #network = load_GPUS(network, torch.load(save_path))
    return network

def LoadDict(model, check):
    model_dict = model.state_dict()
    check_key = list(check.keys())
    #print(check_key)
    model_key = list(model_dict.keys())
    from collections import OrderedDict
    old_model_dict = OrderedDict()
    print(len(model_key))
    print(len(check_key))
    if len(model_key) != len(check_key):
        raise Exception("state dict does not fit model structure")
    for i in range(len(check.keys())):
        old_model_dict[model_key[i]] = check[check_key[i]]
    #model_dict.update(old_model_dict)
    model.load_state_dict(old_model_dict)

    return model





######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    predict = torch.FloatTensor()
    idds = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img_l,img_r,idd = data
        n, c, h, w = img_l.size()
        count += n
        print(count)

        '''
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            f = outputs.data.cpu()
            ff = ff+f
        '''
        input_img_l = Variable(img_l.cuda())
        input_img_r = Variable(img_r.cuda())
        outputs = model(input_img_l,input_img_r)
        m = nn.Sigmoid()
        outputs = m(outputs)
        ff =  outputs.data.cpu()
        predict = torch.cat((predict, ff), 0)
        idds = torch.cat((idds,idd.float()),0)
        #print(predict.shape)
        #print(idds.shape)
    return predict.numpy(),idds.numpy()




######################################################################
# Load Collected data Trained model
print('-------test-----------')

model_structure = ft_efnet(8)


#model = model_structure # test imagenet pretrained model
model = load_network(model_structure)
#print(model)


# Change to test mode
model = model.eval()
model = model.cuda()
# Extract feature
predict,idds = extract_feature(model,test_loader)
'''
print(predict.shape)
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        if predict[i][j] > 0.4:
            predict[i][j] = 1
        else:
            predict[i][j] = 0
'''
import csv

f = open('1.csv','w',encoding='utf-8')

csv_writer = csv.writer(f)

# 3. 构建列表头
csv_writer.writerow(['ID','N','D','G','C','A','H','M','O'])

for i in range(len(idds)):
    csv_writer.writerow([int(idds[i]),predict[i][0],predict[i][1],predict[i][2],predict[i][3],predict[i][4],predict[i][5],predict[i][6],predict[i][7]])

# Save to Matlab for check
#result = {'predict':predict.numpy(),'idds':idds.numpy()}
#scipy.io.savemat('pytorch_result.mat',result)
