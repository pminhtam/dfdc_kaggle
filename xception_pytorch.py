import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
# https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))  
    print(count)
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])   
    print(weight_per_class)
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight   

batch_size=32
num_workers=1
epochs = 20
steps = 0
running_loss = 0
print_every = 5000
checkpoint = "./resnet50_pytorch/"

if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
    
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

# model = models.resnext50_32x4d(pretrained=True)
# model.fc = nn.Sequential(nn.Linear(2048, 1),
#                                  nn.Sigmoid())
# model = model.cuda()
from pytorchcv.model_provider import get_model
model = get_model("xception", pretrained=False)
model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer

class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

        self.p1 = nn.AdaptiveAvgPool2d((1,1))
        self.p2 = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        x1 = self.p1(x)
        x2 = self.p2(x)
        return (x1+x2) * 0.5

model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))

class Head(torch.nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        out = nn.Sigmoid()(out)
        return out

class FCN(torch.nn.Module):
    def __init__(self, base, in_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, 1)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)

net = []
model = FCN(model, 2048)
model = model.cuda()


train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomApply([
                                           transforms.RandomRotation(5),
                                           transforms.RandomAffine(degrees=5,scale=(0.95,1.05))
                                           ], p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                                       
                                       ])
train_data = datasets.ImageFolder('/data/tam/kaggle/extract_raw_img',       
                    transform=train_transforms)


weights = make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes))                                                                
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)) 

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = sampler, num_workers=num_workers)

test_data = datasets.ImageFolder('/data/tam/kaggle/extract_raw_img_test',       
                    transform=train_transforms)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=1)

criterion = nn.BCELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.003)

model.load_state_dict(torch.load("/home/tampm/code/kaggle/resnet50_pytorch/resnet_pytorch_1.pt"))

train_losses, test_losses = [], []
import time
for epoch in range(epochs):
    for inputs, labels in tqdm(trainloader):
        model.train()
#     for inputs, labels in tqdm(testloader):

        steps += 1
#         labels = np.array([labels])
        inputs, labels = inputs.to(device), labels.float().to(device)
#         inputs, labels = inputs.to(device), labels[1].float().to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)[:,0]
        loss = criterion(logps, labels)
#         loss = F.binary_cross_entropy_with_logits(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
#         time.sleep(0.5)
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.float().to(device)
                    logps = model.forward(inputs)[:,0]
                    batch_loss = criterion(logps, labels)
    #                 batch_loss = F.binary_cross_entropy_with_logits(logps, labels)
                    test_loss += batch_loss.item()
#                     print("labels : ",labels)
#                     print("logps  : ",logps)
                    equals = labels == (logps >0.5)
    #                     print("equals   ",equals)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
#                 train_losses.append(running_loss/len(trainloader))
    #             test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
    torch.save(model.state_dict(), os.path.join(checkpoint, 'resnet_pytorch_%d.pt' % epoch))
