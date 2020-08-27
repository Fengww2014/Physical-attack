import os
import sys
import cv2
import glob
import numpy as np
import pdb


IMG_PATH = '../../../advGAN_results/test/attack_v2_193t-193_b3p10/test_latest/images/'
imgs_real = glob.glob(IMG_PATH + '*real_A.png')
imgs_rec = glob.glob(IMG_PATH + '*rec_A.png')
imgs_real = sorted(imgs_real)
imgs_rec = sorted(imgs_rec)
num_imgs = len(imgs_real)
img_real = cv2.imread(imgs_real[0])
height = img_real.shape[0]
width = img_real.shape[1]
noise_level = 0
for i in range(num_imgs): 
    img_real = cv2.imread(imgs_real[i])
    img_rec = cv2.imread(imgs_rec[i])
    
    img_noise = cv2.subtract(img_rec, img_real) 
    noise_level += cv2.norm(img_noise)/(3*height*width)
    #cv2.imshow("img_noise", img_noise)
    #cv2.waitKey(0)
    #cv2.imwrite(imgs_real[i].replace('real_A', 'noise'), img_noise)
noise_level = noise_level/num_imgs
print(noise_level)

