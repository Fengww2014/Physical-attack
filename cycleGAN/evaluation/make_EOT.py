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

read_path = './images/rec_A_imgs'
imgs = glob.glob(read_path + '/*.png')
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


for i in range(num_imgs): 
    resize_para = 0.9
    rotation_para = 0
    img = Image.open(imgs[i])
    img_name = imgs[i].replace(read_path, '')
    #print(img_name)
    img_torch_0 = transforms.ToTensor()(img)
    img_torch_1 = Resize(img_torch_0, resize_para)
    img_torch_2 = Rotation(img_torch_1, rotation_para)
    save_path = read_path+'_s'+str(resize_para)+'r'+str(rotation_para)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print(save_path + img_name)
    save_image(img_torch_2, save_path + img_name)
