import torch
import numpy as np

# output = torch.tensor(np.random.choice([0, 1], [5, 6]))
# target = torch.tensor(np.random.choice([0, 1], [5, 6]))
# print(output)
# print(target)
#target = output
# output = torch.tensor(np.array([[0,1,0],[0,1,0]]))
# target = torch.tensor(np.array([[1, 0,1], [1,0, 1]]))

def accuracy_1875(output, target):  # Tensor:Tensor #size: batchsize252


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_list=[0]*375  #储存每个对的次数
    batchsize = output.size(0)
    assert(batchsize == target.size(0))
    p = torch.chunk(output, 5, 1)  # p[0]–p[6], batchsize36
    t = torch.chunk(target, 5, 1)

    a = np.ones((batchsize, 1), np.dtype('i8'))*5
    ps = torch.from_numpy(a).to(device)
    ts = torch.from_numpy(a).to(device)  # LongTensor, tmp, and will be cut

    for i in range(0, 5):   # the index of max value in every segment
        _, pred = torch.max(p[i], 1)
        ps = torch.cat((ps, pred.resize(batchsize,1)), 1)
        _, pred = torch.max(t[i], 1)
        ts = torch.cat((ts, pred.resize(batchsize,1)), 1)
    sub = torch.LongTensor([1, 2, 3, 4, 5]).to(device)
    ps = torch.index_select(ps, 1, sub)  # LongTensor
    ts = torch.index_select(ts, 1, sub)  # LongTensor

    tspseq = torch.eq(ts, ps)  # ByteTensor
    for i in range(batchsize):
        for j in range(2):
            if tspseq[i][j]==1:
                class_num=ps[i][j]
                acc_list[class_num] += 1
    return acc_list

def accuracy_375(output,target):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batchsize = output.size(0)
    class_num =output.size(1)
    assert(batchsize == target.size(0))

    acc_list = [] # 储存每个对的次数
    for m in range(class_num):
        acc_list.append([0,0])
    
    output_one_hot=(output>0)
    #print(output_one_hot)
    for i in range(batchsize):
        #print(acc_list)
        for j in range(class_num):
            #print(acc_list)
            if output_one_hot[i][j] == 1:
                acc_list[j][0] = acc_list[j][0]+1
                if target[i][j]==1:
                    acc_list[j][1] = acc_list[j][1]+1
    return acc_list
