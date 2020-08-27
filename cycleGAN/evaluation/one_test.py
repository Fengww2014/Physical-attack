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
import math


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

imgs = glob.glob('./images/*png')
#imgs = glob.glob('H:/advGAN_results/fakeA/attack_193pt207_b3p10_fake_A/test_latest/images/*real_B.png')
model = models.vgg16(pretrained=True)#.cuda()  #resnet50/mobilenet_v2/alexnet/vgg16/vgg19
model.eval()
transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])
target_class = 305
imgs = sorted(imgs)
num_imgs = min(200,len(imgs))
corr_count = 0.0
confidence_total = 0.0
img = Image.open(imgs[0])
imgs_torch = transform(img).unsqueeze(0)

img = Image.open(imgs[0])
#print(i,imgs[i])
#img_torch = transform(img).unsqueeze(0)
#print(imgs_torch.shape)

img_torch_0 = transform(img)#.unsqueeze(0)#.cuda()
img_torch_1 = Resize(img_torch_0, 0.8)
img_torch_2 = Rotation(img_torch_1, 0)
img_torch_2 = img_torch_2.unsqueeze(0)
output = model(img_torch_2)

test_index = functional.softmax(output).max(1).indices.numpy()  
confidence_test = functional.softmax(output).max(1).values.detach().numpy()  
least_index = functional.softmax(output).min(1).indices.numpy()  
confidence_least = functional.softmax(output).min(1).values.detach().numpy()  
confidence_cur = functional.softmax(output)[:, target_class]


print('test_index: {0}, confidence_test: {1}, least_index: {2}, confidence_least: {3},  confidence_cur: {4}'.format(test_index, confidence_test, least_index, confidence_least, confidence_cur))
    
