import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

# (1, 0) => target labels 0+2
# (0, 1) => target labels 1
# (1, 1) => target labels 3
train = []
labels = []
for i in range(1000):
    category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
    if category == (1, 0):
        train.append(np.random.uniform(0.1, 1,1874))
        labels.append(np.random.choice([0, 1],100))
    if category == (0, 1):
        train.append(np.random.uniform(0.1, 1,1874))
        labels.append(np.random.choice([0, 1], 100))
    if category == (0, 0):
        train.append(np.random.uniform(0.1, 1,1874))
        labels.append(np.random.choice([0, 1], 100))
print(train[0].__len__())
print(labels[0])

class _classifier(nn.Module):
    def __init__(self, nlabel):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1874, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
        )

    def forward(self, input):
        return self.main(input)


nlabel = len(labels[0])  # => 3
classifier = _classifier(nlabel)

optimizer = optim.Adam(classifier.parameters())
criterion = nn.MultiLabelMarginLoss()

epochs = 5
for epoch in range(epochs):
    losses = []
    for i, sample in enumerate(train):
        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
        #print(labelsv)

        output = classifier(inputv)
       # print(output)
        loss = criterion(output, labelsv.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))
