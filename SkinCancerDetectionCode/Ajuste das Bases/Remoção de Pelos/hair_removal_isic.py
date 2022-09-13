
import cv2
import os 
from pathlib import PurePath
import shutil
import numpy as np


# Path dataset ISIC 2020 and PAD-UFES

path_isic = os.listdir('D:/Bases de Imagens/ISIC2020/ISIC_2020_Training_JPEG')

path_pad_ufes = os.listdir('D:/Bases de Imagens/PAD-UFES/images')


# Path out hair removal images ISIC2020 and PAD-UFES

out_dir_isic = 'D:/Bases de Imagens/ISIC2020/ISIC_2020_hair_removal/'

out_dir_pad_ufes = 'D:/Bases de Imagens/PAD-UFES/images_hair_removal/'

# Aux variables

data_isic = []

data_pad_ufes = []

for name in path_isic:
    d = os.path.join('D:/Bases de Imagens/ISIC2020/ISIC_2020_Training_JPEG/', 
                    name)
    data_isic.append(d)

for name in path_pad_ufes:
    d = os.path.join('D:/Bases de Imagens/PAD-UFES/images', 
                    name)
    data_pad_ufes.append(d)

# Kernel type and size 

kernel_fill = cv2.getStructuringElement(1,(23,23))
kernel_dilatate = cv2.getStructuringElement(2,(3,3))
kernel = cv2.getStructuringElement(2,(21,21))

# Percent images resize

scala = 50 

# aux count
  
i = 0

# Loop remove hair
 
for im in data_isic:
    img = cv2.imread(im, cv2.IMREAD_COLOR)
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel_fill)
    bhg= cv2.GaussianBlur(blackhat,(19,19),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, kernel_dilatate, iterations = 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_non_zero = cv2.countNonZero(mask)
    if num_non_zero > 20000:
        dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
        aux = os.path.split(data_isic[i])
        path = (out_dir_isic + str(aux[1]))
        cv2.imwrite(path, dst)
        i = i+1
    else:
        aux = os.path.split(data_isic[i])
        path = (out_dir_isic + str(aux[1]))
        cv2.imwrite(path, img)
        i=i+1
    

# i =0

# kernel = cv2.getStructuringElement(1,(17,17))

# for im in data_pad_ufes:
    
#     img = cv2.imread(im, cv2.IMREAD_COLOR)
#     grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     blackhat = cv2.dilate(grayScale,kernel,iterations = 100)
#     blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_BLACKHAT, kernel) 
#     #bhg= cv2.GaussianBlur(blackhat,(11,11),cv2.BORDER_DEFAULT)
#     ret,mask = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
#     dst = inpaint.inpaint_biharmonic(img, mask, multichannel=(True))
#     dst = dst/dst.max()
#     dst = 255*dst
#     dst = dst.astype(np.uint8)
#     aux = os.path.split(data_pad_ufes[i])
#     i = i+1
#     path = (out_dir_pad_ufes + str(aux[1]))
#     cv2.imwrite(path, dst)

    

    