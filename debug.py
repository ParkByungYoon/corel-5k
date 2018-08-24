import os
import shutil

test_name_list = list()
with open('./labels/training_label', 'r') as f_test_label:  # 读测试的label文件
    for line in f_test_label.readlines():
        #print(int(line.split(' ')[0]))
        test_name = line.split(' ')[0]
        test_name_list.append(test_name)
#print(test_name_list)

for _,_,a in os.walk('./train/'):
    break
b=[i.split('.')[0] for i in a]
print(set(b)-set(test_name_list))

