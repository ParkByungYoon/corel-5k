import re
from PIL import Image



#选出数量最多的100个标签
def select_100(path):

    #path: label文件的路
    
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
#print(len(select_100('./labels/training_label')))