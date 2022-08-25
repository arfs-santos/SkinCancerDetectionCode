
import cv2
import os 
import numpy as np
from skimage.restoration import inpaint


path_isic = os.listdir('D:/Bases de Imagens/ISIC2020/ISIC_2020_Training_JPEG')

path_pad_ufes = os.listdir('D:/Bases de Imagens/PAD-UFES/images')

out_dir_isic = 'D:/Bases de Imagens/ISIC2020/ISIC_2020_hair_removal/'

out_dir_pad_ufes = 'D:/Bases de Imagens/PAD-UFES/images_hair_removal/'

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


kernel = cv2.getStructuringElement(1,(23,23))
    
i = 0
    
# for im in data_isic:
    
#     img = cv2.imread(im, cv2.IMREAD_COLOR)
#     grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
#     # bhg= cv2.GaussianBlur(blackhat,(11,11),cv2.BORDER_DEFAULT)
#     ret,mask = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
#     dst = cv2.inpaint(img,mask,1,cv2.INPAINT_TELEA)
#     aux = os.path.split(data_isic[i])
#     i = i+1
#     path = (out_dir_isic + str(aux[1]))
#     cv2.imwrite(path, dst)

i =0

kernel = cv2.getStructuringElement(1,(17,17))

for im in data_pad_ufes:
    
    img = cv2.imread(im, cv2.IMREAD_COLOR)
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.dilate(grayScale,kernel,iterations = 100)
    blackhat = cv2.morphologyEx(blackhat, cv2.MORPH_BLACKHAT, kernel) 
    #bhg= cv2.GaussianBlur(blackhat,(11,11),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    dst = inpaint.inpaint_biharmonic(img, mask, multichannel=(True))
    dst = dst/dst.max()
    dst = 255*dst
    dst = dst.astype(np.uint8)
    aux = os.path.split(data_pad_ufes[i])
    i = i+1
    path = (out_dir_pad_ufes + str(aux[1]))
    cv2.imwrite(path, dst)

    

    