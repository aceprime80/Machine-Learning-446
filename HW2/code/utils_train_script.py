import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from hw2_ResNet import ResNet
import struct
import os
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
device = torch.device('mps')
print(torch.has_mps)

root = './hw2_data'
if not os.path.exists(root):
    os.mkdir(root)

normalization = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=normalization, download=True)
test_set = dset.MNIST(root=root, train=False, transform=normalization, download=True)
trainLoader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
# train_set.train_data.to(device)
# train_set.train_labels.i
# test_set.train_data.to(device)
# test_set.train_labels.to(device)

net = ResNet(4)
net.to('mps')
numparams = 0
#print(net.parameters)
for f in net.parameters():
    print(f.size())
    numparams += f.numel()

optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

def test(net, testLoader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for (data,target) in testLoader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        print("Test Accuracy: %f" % (100.*correct/len(testLoader.dataset)))
        return 1 - (correct/len(testLoader.dataset))

test(net, testLoader)
train_acc = []
test_acc = []
for epoch in range(400):
    net.train()
    for batch_idx, (data, target) in enumerate(trainLoader):
        data = data.to(device)
        target = target.to(device)
        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        gn = 0
        for f in net.parameters():
            gn = gn + torch.norm(f.grad)
        #print("E: %d; B: %d; Loss: %f; ||g||: %f" % (epoch, batch_idx, loss, gn))
        optimizer.step()
        optimizer.zero_grad()
    print("Epoch: %d" % (epoch))
    train_acc.append(test(net, trainLoader))
    test_acc.append(test(net, testLoader))
    
    #test(net, testLoader)
#print(train_acc)
#print(test_acc)
fig, ax = plt.subplots(2,1,figsize=(7,7))
fig.tight_layout(pad=2)
epochs = np.linspace(1, 400, 400)
ax[0].plot(epochs, train_acc, 'r-')
ax[0].set_title("Training Error vs # of Epochs")
ax[0].set_ylabel("Training accuracy %")
ax[0].set_xlabel("# of epochs")
ax[1].plot(epochs, test_acc, 'b-')
ax[1].set_title("Testing Error vs # of Epochs")
ax[1].set_ylabel("Test accuracy %")
ax[1].set_xlabel("# of epochs")


plt.show()