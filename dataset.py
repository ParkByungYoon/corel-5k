import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def combine(path):
    #label_file_path:label文件的路径
    a = []
    pattern = re.compile(r'\s\d{1,3}')
    with open(path, 'r') as f:
        line = f.readlines()
        for i in range(len(line)):
            match = pattern.findall(line[i])
            match = list(map(lambda x: int(x), match))
            zeros = [0]*(5-len(match))
            match = match + zeros
            a.append(match)
    return a


class NPSET(Dataset):

    def __init__(self, root, data_transform=None, train=None):
        self.picroot = root
        self.data_transform = data_transform
        if not os.path.exists(self.picroot):
            raise RuntimeError('{} doesnot exists'.format(self.picroot))
        if train:
            jpegs_path = os.path.join(self.picroot, 'train_jpeg/')
            label_path = os.path.join(self.picroot, 'labels/training_label')
        else:
            jpegs_path = os.path.join(self.picroot, 'test_jpeg/')
            label_path = os.path.join(self.picroot, 'labels/test_label')
        imgs = []
        a = 0
        with open(label_path, 'r') as file:
            for line in file.readlines():
                jpeg_name = line.split(' ')[0]+'.jpeg'
                jpeg_path = os.path.join(jpegs_path, jpeg_name)
                image = Image.open(jpeg_path).convert('RGB')
                imgs.append(image)
            a = len(file.readlines())
        self.dataset = imgs
        self.labels = combine(label_path)
        self.len = a


    def code_to_vec(self, p, code):
        def char_to_vec(c):
            y = np.zeros((374,))
            y[c] = 1.0
            return y
        c = np.vstack([char_to_vec(c) for c in code])
        return c.flatten()

    def __getitem__(self, index):
        label, img = self.labels[index], self.dataset[index]
        if self.data_transform is not None:
            img = self.data_transform(img)
        labelarray = self.code_to_vec(1, label)
        return (img, labelarray)

    def __len__(self):
        return self.len


