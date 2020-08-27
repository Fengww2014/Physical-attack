import cv2
from skimage.measure import compare_ssim
import glob
import os



if __name__ == "__main__":
    WSI_MASK_PATH = 'images/ex_4_7_2/'#存放图片的文件夹路径
    paths_A = glob.glob(os.path.join(WSI_MASK_PATH, '*real_A.png'))
    paths_A.sort()
    ssim_sum = 0
    i = 0
    for pathA in paths_A:
        img1= cv2.imread(pathA, 0)   
        pathB = pathA.replace('real_A.', 'fake_B.')
        img2 = cv2.imread(pathB, 0)
        #print(compare_ssim(img1, img2, multichannel=True))
        ssim_sum += compare_ssim(img1, img2, multichannel=True)
        i += 1
    ssim = ssim_sum / i
    print(ssim)
