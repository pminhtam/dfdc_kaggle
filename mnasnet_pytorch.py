import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

batch_size=16
num_workers=1
epochs = 30
steps = 0
running_loss = 0
print_every = 5000
checkpoint = "./mnasnet_pytorch/"

if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)
    
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

model = models.mnasnet1_0(pretrained=True).cuda()
model.classifier = nn.Sequential(nn.Linear(1280, 1),
                                 nn.Sigmoid())
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
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers)

criterion = nn.BCELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.003)

model.load_state_dict(torch.load("/home/tampm/code/kaggle/mnasnet_pytorch/mnasnet_pytorch_2.pt"))

train_losses, test_losses = [], []
import time
text_writer = open(os.path.join(checkpoint, 'train.csv'), 'a')
model.train()
for epoch in range(epochs):
    for inputs, labels in tqdm(trainloader):
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
        time.sleep(0.05)
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
            text_writer.write('Epoch %d, Train loss %.4f, Test loss %.4f, Test accuracy  %.4f \n'   % (epoch, running_loss/print_every, test_loss/len(testloader), accuracy/len(testloader)))
            text_writer.flush()
            
            running_loss = 0
            model.train()
    torch.save(model.state_dict(), os.path.join(checkpoint, 'mnasnet_pytorch_%d.pt' % epoch))
