import cv2                   #导入opencv库
import glob
import os

WSI_MASK_PATH = './images/ones/scan_fakeB_pages/'#存放图片的文件夹路径
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
for path in paths:
    os.rename(path,path.replace('页面_',''))

paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
paths.sort()
store_folder = WSI_MASK_PATH.replace('_pages', '_imgs')
if not os.path.exists(store_folder):
    os.makedirs(store_folder)
#1241*1754
w = 1240
h = 1753
h0 = 165
h = 405

n=0
dh=15
for path in paths:
    print(path)
    img= cv2.imread(path)     

    img_split_l1 = img[h0:h0+h, w-1034: w-653, :]
    #img_split_l1 = cv2.resize(img_split_l1, (256, 256)) 
    cv2.imwrite(store_folder + str(6*n+1) + '.png', img_split_l1)
    img_split_r1 = img[h0:h0+h, 653:1034, :]
    #img_split_r1 = cv2.resize(img_split_r1, (256, 256)) 
    cv2.imwrite(store_folder + str(6*n+2) + '.png', img_split_r1)
    img_split_l2 = img[h0+h+dh: h0+2*h+dh, w-1034: w-653, :]
    #img_split_l2 = cv2.resize(img_split_l2, (256, 256)) 
    cv2.imwrite(store_folder + str(6*n+3) + '.png', img_split_l2)
    img_split_r2 = img[h0+h+dh: h0+2*h+dh, 653:1034, :]
    #img_split_r2 = cv2.resize(img_split_r2, (256, 256)) 
    cv2.imwrite(store_folder + str(6*n+4) + '.png', img_split_r2)
    img_split_l3 = img[h0+2*h+dh*2: h0+3*h+dh*2, w-1034: w-653, :]
    #img_split_l3 = cv2.resize(img_split_l3, (256, 256)) 
    cv2.imwrite(store_folder + str(6*n+5) + '.png', img_split_l3)
    img_split_r3 = img[h0+2*h+dh*2: h0+3*h+dh*2, 653:1034, :]
    #img_split_r3 = cv2.resize(img_split_r3, (256, 256)) 
    cv2.imwrite(store_folder + str(6*n+6) + '.png', img_split_r3)


    n += 1                         
    
