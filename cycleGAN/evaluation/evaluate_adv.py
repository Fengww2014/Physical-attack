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
# recs_A = glob.glob('../results/' + ex + '/test_latest/images/*rec_A.png')
# model = models.alexnet(pretrained=True).cuda() #alexnet
model = models.vgg16(pretrained=True).cuda() #vgg16
model.eval()
transform = transforms.Compose([transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))])


real_As = sorted(reals_A)
fake_Bs = sorted(fakes_B)
# rec_As = sorted(recs_A)
num_imgs = len(real_As)
corr_count_real_A = 0.0
corr_count_fake_B = 0.0
corr_count_rec_A = 0.0
real_A_img = Image.open(real_As[0])
real_A_imgs_torch = transform(real_A_img).unsqueeze(0).cuda()
fake_B_img = Image.open(fake_Bs[0])
fake_B_imgs_torch = transform(fake_B_img).unsqueeze(0).cuda()
# rec_A_img = Image.open(rec_As[0])
# rec_A_imgs_torch = transform(rec_A_img).unsqueeze(0).cuda()

for i in range(1, num_imgs):
    real_A_img = Image.open(real_As[i])
    real_A_img_torch = transform(real_A_img).unsqueeze(0).cuda()
    real_A_imgs_torch = torch.cat( (real_A_imgs_torch, real_A_img_torch) ,dim = 0)
    fake_B_img = Image.open(fake_Bs[i])
    fake_B_img_torch = transform(fake_B_img).unsqueeze(0).cuda()
    fake_B_imgs_torch = torch.cat( (fake_B_imgs_torch, fake_B_img_torch) ,dim = 0)
    # rec_A_img = Image.open(rec_As[i])
    # rec_A_img_torch = transform(rec_A_img).unsqueeze(0).cuda()
    # rec_A_imgs_torch = torch.cat( (rec_A_imgs_torch, rec_A_img_torch) ,dim = 0)


output_real_A = model(real_A_imgs_torch)
output_fake_B = model(fake_B_imgs_torch)
# output_rec_A = model(rec_A_imgs_torch)
test_index_real_A = functional.softmax(output_real_A).max(1)[1].cpu().detach().numpy()  
test_index_fake_B = functional.softmax(output_fake_B).max(1)[1].cpu().detach().numpy()  
# test_index_rec_A = functional.softmax(output_rec_A).max(1)[1].cpu().detach().numpy() 
confidence_cur_real_A = functional.softmax(output_real_A)[:, target_class]
confidence_cur_fake_B = functional.softmax(output_fake_B)[:, target_class]
# confidence_cur_rec_A = functional.softmax(output_rec_A)[:, target_class]
confidence_total_real_A = sum(confidence_cur_real_A)
confidence_total_fake_B = sum(confidence_cur_fake_B)
# confidence_total_rec_A = sum(confidence_cur_rec_A)
for j in range(num_imgs):
    if test_index_real_A[j] == target_class:
        corr_count_real_A += 1
    if test_index_fake_B[j] == target_class:
        corr_count_fake_B += 1
    # if test_index_rec_A[j] == target_class:
    #     corr_count_rec_A += 1
acc_real_A = corr_count_real_A/num_imgs
acc_fake_B = corr_count_fake_B/num_imgs
# acc_rec_A = corr_count_rec_A/num_imgs
confidence_real_A = confidence_total_real_A/num_imgs
confidence_fake_B = confidence_total_fake_B/num_imgs
# confidence_rec_A = confidence_total_rec_A/num_imgs


print('real_A acc: {0}, real_A confidence: {1}\r\n'.format(acc_real_A, confidence_real_A))
print('fake_B acc: {0}, fake_B confidence: {1}\r\n'.format(acc_fake_B, confidence_fake_B))
# print('rec_A acc: {0}, rec_A confidence: {1}\r\n'.format(acc_rec_A, confidence_rec_A))
