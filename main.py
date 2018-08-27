# coding=utf-8
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from nets import VGG
from tool import dataset, get_acc, process_jpeg, select100
from tool.data_up import data_augmentation
# 定义是否使用GPU


# #先一步处理图片,加高斯噪声和改变颜色
# process_jpeg.change_jpeg('/home/hbw/corel-5k/train_jpeg/',
#                          '/home/hbw/corel-5k/process_train_jpeg/')
# process_jpeg.change_jpeg(
#                         '/home/hbw/corel-5k/test_jpeg/', 
#                         '/home/hbw/corel-5k/process_test_jpeg/')


# print(train_dataset.__getitem__(0))
#train_datalodar=Data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

#划分交叉验证
def getitems(trainset, begin, end):
    item = []
    num = begin
    for i in range(0, end - begin):
        item.append(trainset.__getitem__(num))
        num += 1
    return item


def tra_and_val(trainset, begin, kfold):
    valid, train = [], []
    num = int(trainset.__len__() / kfold)  # 每一份的个数
    # print(trainset.__len__())
    valid = getitems(trainset, begin, begin + num)  # 验证集只有一份
    train = getitems(trainset, 0, begin)
    train1 = getitems(trainset, begin + num, trainset.__len__())
    train[0:0] = train1
    return valid, train



def train(epoch):
    global j
    global best_acc
    if j==0:
        net=VGG.VGG19().to(device)
    else:
        net=torch.load('VGGnet.pkl')
    for i in range(kfold):
        valid, train = tra_and_val(trainset=train_dataset, begin=int(
            i * train_dataset.__len__() / kfold), kfold=kfold)  # 训练集与验证集
        trainloader = Data.DataLoader(
            dataset=train,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2)
        validloader = Data.DataLoader(
            dataset=valid,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2)
        # print(validloader)
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        loss_func = torch.nn.MultiLabelMarginLoss()


        list_100 = select100.select_100('./labels/training_label')
        list_label=[label[0] for label in list_100]   #前100个标签
        list_100_num=sum([the_tuple[1] for the_tuple in list_100 ])   #数量前100标签的个数和

        train_correct_list=[0]*375   #训练时正确的标签总个数
        valid_correct_list=[0]*375  #验证时正确的标签总个数
        print(7)
        for m, (batch_x, batch_y) in enumerate(trainloader):  #训练
            print(batch_y.type(), batch_x.type())
            x = batch_x.to(device)
            y = batch_y.to(device)
            out = net(x)

            acc_list=get_acc.accuracy(out, y)  #返回这个batch_size里面预测对的总个数
            for p,q in enumerate(acc_list):
                if p in list_label:
                    train_correct_list[p] += q
            print(out.type())
            y = y.long()
            loss = loss_func(out, y)
            print(9)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = sum(train_correct_list)/list_100_num
        
        for input1, label in validloader:
            input1 = input1.to(device)
            label = label.to(device)
            out = net(input1)

            acc_list = get_acc.accuracy(out, label)  # 返回这个batch_size里面预测对的总个数
            for p, q in enumerate(acc_list):
                if p in list_label:
                    valid_correct_list[p] += q
        valid_acc = sum(valid_correct_list)/list_100_num
        if valid_acc > best_acc:
            best_acc =valid_acc
            torch.save(net, 'VGGnet.pkl')
        print('Epoch: ', epoch, '| train loss:%.4f' % loss, '| train accuracy: %.4f' % train_acc,
              '| valid accuracy: %.4f' % valid_acc, '| best accuracy: %.4f' % best_acc)
    j+=1


# for epoch in range(EPOCH):
if __name__ == '__main__':
    print(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(2)
    EPOCH=50
    LR=0.0001
    BATCH_SIZE=4
    kfold=9
    best_acc=0
    root='./'
    train_transform=transforms.Compose([
    transforms.RandomCrop((128,192),padding=8),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5049, 0.5524, 0.4342), (0.3692, 0.3545, 0.3728)
    )
    ])
    print(3)
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4989, 0.5541, 0.4327), (0.3666, 0.3544, 0.3725)
        )
    ])
    print(4)
    #训练集
    train_dataset = dataset.NPSET(root=root, data_transform=train_transform, train=True)
    print(5)
    #测试集
    test_dataset = dataset.NPSET(root=root, data_transform=test_transform, train=False)
    test_datalodar = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,shuffle=True, num_workers=2)

    print(6)

    j=0  #判断是初始化还是读取模型
    train(1)
