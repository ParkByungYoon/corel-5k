from PIL import Image
import torch
from torchvision import transforms
import math



def regular(label_file_path, img_dir):
    def std(i, mean):
        sq_sum = 0.
        for num in range(len(lines)):
            path = img_dir + lines[num].split(' ')[0] + '.jpeg'
            img = Image.open(path, 'r').convert('RGB')
            trans = transforms.ToTensor()
            tensor = trans(img)
            height = tensor[i].size()[0]
            weight = tensor[i].size()[1]
            for j in range(height):
                for k in range(weight):
                    sq_sum +=math.pow(tensor[i][j][k]-mean, 2)
        std = math.sqrt(sq_sum/(height*weight*len(lines)))
        return std 
    r_mean = r_std = 0.
    g_mean = g_std = 0.
    b_mean = b_std = 0.    
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for num in range(len(lines)):
            path = img_dir + lines[num].split(' ')[0] + '.jpeg'
            img = Image.open(path).convert('RGB')
            trans = transforms.ToTensor()
            tensor = trans(img)
            r_mean += tensor[0].mean()
            g_mean += tensor[1].mean()
            b_mean += tensor[2].mean()
        r_mean = r_mean/len(lines)
        g_mean = g_mean/len(lines)
        b_mean = b_mean/len(lines)
        return [r_mean, g_mean, b_mean], [std(0, r_mean), std(1, g_mean), std(2,b_mean)]



if __name__ == '__main__':
    train_label_path = './labels/training_label'
    train_img_dir = './process_train_jpeg/'
    test_label_path = './labels/test_label'
    test_img_dir = './process_test_jpeg/'
    train = regular(train_label_path, train_img_dir)
    print('train:\n')
    print(train)
    print('\n\n')
    test = regular(test_label_path, test_img_dir)
    print('test:\n')
    print(test)