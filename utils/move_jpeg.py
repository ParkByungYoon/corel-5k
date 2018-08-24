import os
import shutil

for root,dirs,files in os.walk('./images/'):
    break
for the_dir in dirs:
    the_path = os.path.join('./images/',the_dir)
    for dir_root, dir_dirs, dir_files in os.walk(the_path):
        for i in dir_files:
            jpeg_path = os.path.join(the_path,i)
            shutil.move(jpeg_path, './train5/')
