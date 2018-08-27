import os
import shutil

test_name_list=list()
with open('./labels/test_label','r') as f_test_label:  #读测试的label文件
    for line in f_test_label.readlines():
        #print(int(line.split(' ')[0]))
        test_name=line.split(' ')[0]
        test_name_list.append(test_name)
#print(test_name_list.__len__())
for images_root, images_dirs, images_files in os.walk('./images'):
    break

for the_dir in images_dirs:
    all_path = os.path.join('./images/', the_dir)
    for root, dirs, files in os.walk(all_path):
        break
    for the_file in files:
        jpeg_name=os.path.splitext(the_file)[0]
        if jpeg_name in test_name_list:
            now_path=os.path.join(all_path,the_file)
            shutil.move(now_path,'./test/')


