import torch
import numpy as np

# output = torch.tensor(np.random.randint(1, 100, [120, 1875]))
# target = torch.tensor(np.random.randint(1, 100, [120, 1875]))
# #target = output

def accuracy(output, target):  # Tensor:Tensor #size: batchsize252

    acc_list=[0]*375  #储存每个对的次数
    batchsize = output.size(0)
    assert(batchsize == target.size(0))
    p = torch.chunk(output, 5, 1)  # p[0]–p[6], batchsize36
    t = torch.chunk(target, 5, 1)

    a = np.ones((batchsize, 1), np.dtype('i8'))*5
    ps = torch.from_numpy(a)
    ts = torch.from_numpy(a)  # LongTensor, tmp, and will be cut

    for i in range(0, 5):   # the index of max value in every segment
        _, pred = torch.max(p[i], 1)
        ps = torch.cat((ps, pred.resize(batchsize,1)), 1)
        _, pred = torch.max(t[i], 1)
        ts = torch.cat((ts, pred.resize(batchsize,1)), 1)
    sub = torch.LongTensor([1, 2, 3, 4, 5])
    ps = torch.index_select(ps, 1, sub)  # LongTensor
    ts = torch.index_select(ts, 1, sub)  # LongTensor

    tspseq = torch.eq(ts, ps)  # ByteTensor
    for i in range(batchsize):
        for j in range(5):
            if tspseq[i][j]==1:
                class_num=ps[i][j]
                acc_list[class_num] += 1
    return acc_list
