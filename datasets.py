import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from functional import RandomCrop, CenterCrop, RandomFlip, RandomRotate, RandomTrans, AddGussianNoise, AddSaltPepperNoise




class ImageDataset(Dataset):
    def __init__(self, root, transforms_c=None, transforms_=None, transforms_m=None):
        self.transform = transforms.Compose(transforms_)
        self.transform_m = transforms.Compose(transforms_m)
        self.transforms_c = transforms.Compose(transforms_c)

        self.files = []


        listdirs = os.listdir(root)
        for listd in tqdm(listdirs):
            files = sorted(glob.glob(root + '/' + listd + '/*.*'))
            self.files += files
        # self.files = sorted(glob.glob(root + '/images/*.*'))
        # self.files = sorted(glob.glob(root + '/*.*'))
    def __getitem__(self, index):
        # name = int(self.files[index].split('/')[-1].split('.')[0])
        img = Image.open(self.files[index]).convert('RGB')
        mark = Image.open("data/airplane.png").convert('P')

        return self.transform(img), self.transform_m(mark)

    def __len__(self):
        return len(self.files)#,len(self.files1)

class ImageDataset_test(Dataset):
    def __init__(self, root, transforms_=None, transforms_m=None):
        self.transform = transforms.Compose(transforms_)
        self.transform_m = transforms.Compose(transforms_m)
        # self.files = sorted(glob.glob(root + '/images/*.*'))
        self.files = sorted(glob.glob(root + '/*.*'))


        # self.files = []
        # # root = os.path.join(root, "images")
        # listdirs = os.listdir(root)
        # for listd in tqdm(listdirs):
        #     files = sorted(glob.glob(root + '/' + listd + '/*.*'))
        #     self.files += files
    def __getitem__(self, index):
        # name = int(self.files[index].split('/')[-1].split('.')[0])
        name = self.files[index].split('/')[-1]
        img = Image.open(self.files[index]).convert('RGB')
        mark = Image.open("data/airplane.png").convert('P')

        return self.transform(img), self.transform_m(mark), name

    def __len__(self):
        # print(len(self.files))
        return len(self.files)

# Configure dataloaders

def Get_dataloader(path,batch, img_size):
    #Image.BICUBIC
    # transforms_ = [ transforms.Resize((32, 32)),
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transforms_c = [
        transforms.Resize((int(img_size), int(img_size))),
        transforms.RandomChoice([RandomCrop(int(img_size)), CenterCrop(int(img_size))])
    ]
    transforms_ = [  #RandomRotate(32),  # 32
        transforms.Resize((int(img_size), int(img_size))),
        transforms.RandomChoice([RandomFlip(), RandomRotate(60), AddSaltPepperNoise(), AddGussianNoise()]),
        #transforms.RandomChoice([RandomFlip(), RandomRotate(60)]),
        # transforms.RandomChoice([AddSaltPepperNoise(), AddGussianNoise()]),
        # AddSaltPepperNoise(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transforms_m = [
        RandomTrans(),
        transforms.Resize((int(img_size), int(img_size))),
        # RandomRotate(60),
        transforms.ToTensor()
            ]
    train_dataloader = DataLoader(
        ImageDataset(path, transforms_c=transforms_c, transforms_=transforms_, transforms_m=transforms_m),
        batch_size=batch, shuffle=True, num_workers=8, drop_last=True)
    return train_dataloader

def Get_dataloader_test(path, batch, img_size, shuffle):
    transforms_ = [
                transforms.Resize((int(img_size), int(img_size))),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    transforms_m = [transforms.Resize((int(img_size), int(img_size))),
                # RandomRotate(60),
                # RandomTrans(),
                transforms.ToTensor()
                ]
    test_dataloader = DataLoader(
        ImageDataset_test(path, transforms_=transforms_, transforms_m=transforms_m),
        batch_size=batch, shuffle=shuffle, num_workers=8, drop_last=False)

    return test_dataloader


