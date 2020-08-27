import numpy as np
import cv2
from scipy import io
import glob

def getNPS(img):
    mat_data = io.loadmat('printable_vals.mat')
    #print(mat_data.keys())
    printable_vals_rgb = mat_data['printable_vals']
    printable_vals = printable_vals_rgb[...,::-1]
    #print(printable_vals.shape)

    def NPS_pixel(p):
        #print('pixel: ', p)
        p = [int(p[0]), int(p[1]), int(p[2])]
        tmp = np.tile(p, (printable_vals.shape[0],1)) 
        tmp2 = printable_vals - tmp
        tmp3 = np.multiply(tmp2, tmp2)
        tmp4 = np.sum(tmp3, axis=1)
        tmp5 = min(tmp4)   
        #print('NPS_pixel: ', tmp5)
        return tmp5

    NPS_matrix = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i,j,:]
            if sum(pixel) == 0:
                #print(pixel)
                continue
            NPS_matrix[i,j] = NPS_pixel(pixel)
    nps =  sum(sum(NPS_matrix))/(img.shape[0]*img.shape[1])
    #print(nps)
    return nps

if __name__ == '__main__':
    imgs = glob.glob('../images/rescan_imgs/*.png')
    imgs = sorted(imgs)
    num_imgs = len(imgs)
    NPS_total = 0.0
    for i in range(num_imgs):
        image = cv2.imread(imgs[i], 1)
        NPS_total += getNPS(image)
    NPS = NPS_total/num_imgs
    print('NPS: {0}'.format(NPS))
    