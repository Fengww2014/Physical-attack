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

 
ex = sys.argv[1] 
target_class = int(sys.argv[2])

if target_class < 0:
    target_class = - target_class

reals_A = glob.glob('../results/' + ex + '/test_latest/images/*real_A.png')
fakes_B = glob.glob('../results/' + ex + '/test_latest/images/*fake_B.png')

model = models.alexnet(pretrained=True).cuda() #alexnet
model.eval()
transform = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])


real_As = sorted(reals_A)
fake_Bs = sorted(fakes_B)

num_imgs = len(real_As)
corr_count_real_A = 0.0
corr_count_fake_B = 0.0

real_A_img = Image.open(real_As[0])
real_A_imgs_torch = transform(real_A_img).unsqueeze(0).cuda()
fake_B_img = Image.open(fake_Bs[0])
fake_B_imgs_torch = transform(fake_B_img).unsqueeze(0).cuda()


for i in range(1, num_imgs):
    real_A_img = Image.open(real_As[i])
    real_A_img_torch = transform(real_A_img).unsqueeze(0).cuda()
    real_A_imgs_torch = torch.cat( (real_A_imgs_torch, real_A_img_torch) ,dim = 0)
    fake_B_img = Image.open(fake_Bs[i])
    fake_B_img_torch = transform(fake_B_img).unsqueeze(0).cuda()
    fake_B_imgs_torch = torch.cat( (fake_B_imgs_torch, fake_B_img_torch) ,dim = 0)



output_real_A = model(real_A_imgs_torch)
output_fake_B = model(fake_B_imgs_torch)

test_index_real_A = functional.softmax(output_real_A).max(1)[1].cpu().detach().numpy()  
test_index_fake_B = functional.softmax(output_fake_B).max(1)[1].cpu().detach().numpy()  

confidence_cur_real_A = functional.softmax(output_real_A)[:, target_class]
confidence_cur_fake_B = functional.softmax(output_fake_B)[:, target_class]

confidence_total_real_A = sum(confidence_cur_real_A)
confidence_total_fake_B = sum(confidence_cur_fake_B)

for j in range(num_imgs):
    if test_index_real_A[j] == target_class:
        corr_count_real_A += 1
    if test_index_fake_B[j] == target_class:
        corr_count_fake_B += 1

acc_real_A = corr_count_real_A/num_imgs
acc_fake_B = corr_count_fake_B/num_imgs

confidence_real_A = confidence_total_real_A/num_imgs
confidence_fake_B = confidence_total_fake_B/num_imgs


print('real_A acc: {0}, real_A confidence: {1}\r\n'.format(acc_real_A, confidence_real_A))
print('fake_B acc: {0}, fake_B confidence: {1}\r\n'.format(acc_fake_B, confidence_fake_B))

