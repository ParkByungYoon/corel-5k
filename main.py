# coding=utf-8
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import torch.nn as nn
from torch.autograd import Variable
from tool import dataset
from tool.data_up import data_augmentation
from tool import process_jpeg

LR=0.0001
BATCH_SIZE=128
root='./'

#先一步处理图片,加高斯噪声和改变颜色
process_jpeg.change_jpeg('/home/hbw/corel-5k/train_jpeg/',
                         '/home/hbw/corel-5k/process_train_jpeg/')
process_jpeg.change_jpeg(
                        '/home/hbw/corel-5k/test_jpeg/', 
                        '/home/hbw/corel-5k/process_test_jpeg/')

train_transform=transforms.Compose([
    transforms.RandomCrop((128,192),padding=8),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(

    # )
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(

    # )
])

#训练集
train_dataset = dataset.NPSET(root=root, data_transform=train_transform, train=True)
train_datalodar=Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#测试集
test_dataset = dataset.NPSET(root=root, data_transform=test_transform, train=False)
test_datalodar = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)
