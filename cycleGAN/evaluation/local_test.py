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


imgs = glob.glob('./images/stop/0stop*')
#imgs = glob.glob('H:/advGAN_results/fakeA/attack_193pt207_b3p10_fake_A/test_latest/images/*real_B.png')
model = models.vgg16(pretrained=True)#.cuda()  #resnet50/mobilenet_v2/alexnet
model.eval()
transform = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))])

target_class = 919
imgs = sorted(imgs)
num_imgs = min(2,len(imgs))
corr_count = 0.0
confidence_total = 0.0
img = Image.open(imgs[0])
imgs_torch = transform(img).unsqueeze(0)
for i in range(1, num_imgs): 
    img = Image.open(imgs[i])
    #print(i,imgs[i])
    img_torch = transform(img).unsqueeze(0)
    imgs_torch = torch.cat( (imgs_torch, img_torch) ,dim = 0)
    #print(imgs_torch.shape)

output = model(imgs_torch)

test_index = functional.softmax(output).max(1).indices.numpy()  
confidence_test = functional.softmax(output).max(1).values.detach().numpy()  
confidence_cur = functional.softmax(output)[:, target_class]
confidence_total = sum(confidence_cur)
for j, cur_index in enumerate(test_index):
    if cur_index == target_class:
        corr_count += 1
    print(j, imgs[j], test_index[j], confidence_test[j],'\r\n')
acc = corr_count/num_imgs
confidence = confidence_total/num_imgs

print('acc: {0}, confidence: {1}'.format(acc, confidence))
    
