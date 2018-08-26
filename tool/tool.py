import re
from PIL import Image


# file_path = './core5k/labels/training_label'
# img_path = 'core5k/images/173000/173015.jpeg'



#选出数量最多的100个标签
def select(path):

    #----------------

    #path: label文件的路径

    #------------------
    
    with open(path,'r') as f:
        pattern = re.compile(r'\s\d{1,3}')
        labels = [0]*375
        line = f.readlines()
        for i in range(len(line)):
            match = pattern.findall(line[i])
            for x in match:
                labels[int(x)] += 1
        ran = list(range(375))
        dicts = dict(zip(ran, labels))
        f.close()
        res = sorted(dicts.items(), key=lambda d: d[1], reverse=True)
        return res[:100]




#以列表形式输出图片的标签组合
def combine(path):

    #-----------------

    #label_file_path:label文件的路径

    #------------------

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



# if __name__ == '__main__':
#     print(select(path))
