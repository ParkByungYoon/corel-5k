import os
from tool.data_up.data_augmentation import DataAugmentation

def change_jpeg(root,save_path):
    for _, no, files in os.walk(root):
        break
    for jpeg_name in files:
        all_path=os.path.join(root,jpeg_name)
        image = DataAugmentation.openImage(all_path)
        image = DataAugmentation.randomColor(image)
        image = DataAugmentation.randomGaussian(image)
        DataAugmentation.saveImage(image, os.path.join(save_path, jpeg_name))
