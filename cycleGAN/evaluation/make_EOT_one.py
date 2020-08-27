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
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import math

path = './images/make_EOT'
imgs = glob.glob(path + '/*.png')
imgs = sorted(imgs)
num_imgs = min(100, len(imgs))


def Rotation(img_torch, degree):
    angle = degree*math.pi/180
    theta = torch.tensor([
        [math.cos(angle),math.sin(-angle),0],
        [math.sin(angle),math.cos(angle) ,0]
    ], dtype=torch.float) + (torch.rand(2,3)-0.5)/10
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
    ], dtype=torch.float) + (torch.rand(2,3)-0.5)/10
    grid = functional.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    output = functional.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    #plt.imshow(new_img_torch.numpy().transpose(1,2,0))
    #plt.show()
    return new_img_torch 

def Fileter(img_torch):
    color_filter  = Variable((torch.eye(3)+0.05*torch.randn(3,3)), requires_grad=True)
    filted = torch.zeros(1,3,256,256)
    for k in range(3):
        processed_filter = torch.unsqueeze(color_filter[k,:], 0)
        processed_filter = torch.unsqueeze(processed_filter, 2)
        processed_filter = torch.unsqueeze(processed_filter, 3)
        filted[:,k,:,:] = torch.sum(processed_filter.repeat(1,1,256,256).mul(img_torch), dim=1, keepdim=True)
    return filted

for i in range(num_imgs): 
    resize_para = 0.9
    rotation_para = 0
    img = Image.open(imgs[i])
    img_base_name = imgs[i].replace(path+'\\', '').replace('.png','')
    img_torch_0 = transforms.ToTensor()(img)
    img_filterd_torch = Fileter(img_torch_0)
    img_name = img_base_name + '_f' + '.png'
    save_image(img_filterd_torch, path+'\\' + img_name)
    if 0:
        for j in range(15):
            resize_para = 1 - 0.02 * j
            for k in range(15):
                rotation_para = 30 - 4 * k
                img_torch_1 = Resize(img_torch_0, resize_para)
                img_torch_2 = Rotation(img_torch_1, rotation_para)
                img_name = img_base_name + '_s'+str(resize_para)+'r'+str(rotation_para) + '.png'
                save_image(img_torch_2, path+'\\' + img_name)
    if 0:
        for n in range(74):
            img_name = img_base_name + '_n'+ str(n) + '.png'
            img_torch_0 = transforms.ToTensor()(img) + torch.randn_like(img_torch_0) * 0.05
            save_image(img_torch_0, path+'\\' + img_name)
