import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename, label_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        self.img_filename_l = [self.img_filename[i] for i in range(len(self.img_filename)) if i % 2 == 0]
        self.img_filename_r = [self.img_filename[i] for i in range(len(self.img_filename)) if i % 2 == 1]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        self.label = labels

        print(len(self.img_filename_l))
        print(len(self.img_filename_r))
        print(len(self.label))


    def __getitem__(self, index):
        img_l = Image.open(os.path.join(self.img_path, self.img_filename_l[index]))
        img_l = img_l.convert('RGB')

        img_r = Image.open(os.path.join(self.img_path, self.img_filename_r[index]))
        img_r = img_r.convert('RGB')

        if self.transform is not None:
            img_l = self.transform(img_l)
            img_r = self.transform(img_r)
        label = torch.from_numpy(self.label[index])
        return img_l, img_r, label
    def __len__(self):
        return len(self.img_filename_l)


class DatasetProcessingTest(Dataset):
    def __init__(self, data_path, img_path, img_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        self.img_filename_l = [self.img_filename[i] for i in range(len(self.img_filename)) if i % 2 == 0]
        self.img_filename_r = [self.img_filename[i] for i in range(len(self.img_filename)) if i % 2 == 1]
        self.img_id_l = [self.img_filename_l[i].split('_')[0] for i in range(len(self.img_filename_l))]
        self.img_id_r = [self.img_filename_r[i].split('_')[0] for i in range(len(self.img_filename_r))]
        fp.close()
        # reading labels from file


    def __getitem__(self, index):
        img_l = Image.open(os.path.join(self.img_path, self.img_filename_l[index]))
        img_l = img_l.convert('RGB')

        img_r = Image.open(os.path.join(self.img_path, self.img_filename_r[index]))
        img_r = img_r.convert('RGB')
        
        idd = float(self.img_id_l[index])

        if self.transform is not None:
            img_l = self.transform(img_l)
            img_r = self.transform(img_r)
        return img_l, img_r, idd
    def __len__(self):
        return len(self.img_filename_l)

