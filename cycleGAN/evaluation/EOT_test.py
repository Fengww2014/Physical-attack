import os
import sys
from PIL import Image
import glob
import numpy as np
import pdb
import torch
from torch.nn import functional
from torch.autograd import Variable
from torchvision import models
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math


#imgs = glob.glob('./images/attack_193pt207_b3p10/photo_fake_B_imgs/*.jpg')
imgs = glob.glob('H:/advGAN_results/fakeA/attack_193pt207_b3p10_fake_A/test_latest/images/*fake_A.png')
model = models.alexnet(pretrained=True)#.cuda()  #resnet50/mobilenet_v2/alexnet
model.eval()
transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])
target_class = 207
imgs = sorted(imgs)
num_imgs = min(100,len(imgs))
corr_count = 0.0
confidence_total = 0.0

def Rotation(img_torch, degree):
    angle = degree*math.pi/180
    theta = torch.tensor([
        [math.cos(angle),math.sin(-angle),0],
        [math.sin(angle),math.cos(angle) ,0]
    ], dtype=torch.float)
    grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    output = functional.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    #plt.imshow(new_img_torch.numpy().transpose(1,2,0))
    #plt.show()
    return new_img_torch


def Resize(img_torch, times):  
    tmp = 1.0/times 
    theta = torch.tensor([
        [tmp, 0 ,0],
        [0 ,tmp ,0]
    ], dtype=torch.float)
    grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    output = functional.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    #plt.imshow(new_img_torch.numpy().transpose(1,2,0))
    #plt.show()
    return new_img_torch


for i in range(num_imgs): 
    img = Image.open(imgs[i])
    img_torch_0 = transform(img)#.unsqueeze(0)#.cuda()
    #img_torch_0 = transforms.ToTensor()(img)
    img_torch_1 = Resize(img_torch_0, 0.5)
    img_torch_2 = Rotation(img_torch_0, 0)
    imgs_torch = torch.stack( (img_torch_0, img_torch_1, img_torch_2) ,dim = 0)
    output = model(imgs_torch)
    #print(functional.softmax(output).max(1))
    test_index = functional.softmax(output).max(1).indices.numpy()  
    confidence_test = functional.softmax(output).max(1).values.detach().numpy()  
    confidence_cur = functional.softmax(output)[:, target_class]
    #print(confidence_cur)
    confidence_total += sum(confidence_cur)
    for j, cur_index in enumerate(test_index):
        if cur_index == target_class:
            corr_count += 1
        #print(imgs[i], j, test_index[j], confidence_test[j],'\r\n')

print(corr_count, num_imgs)
acc = corr_count/(num_imgs*3)
confidence = confidence_total/(num_imgs*3)

print('acc: {0}, confidence: {1}'.format(acc, confidence))
    
