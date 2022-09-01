
import cv2
import os 

#IMAGE ACQUISITION

#Input image

path_isic_maligno = os.listdir('D:/Bases de Imagens/ISIC2020/malign')

path_isic_benigno = os.listdir('D:/Bases de Imagens/ISIC2020/benign')

data_malign=[]

data_benign=[]


out_dir_mel = 'D:/Bases de Imagens/ISIC2020/malign_hair_removal/'
out_dir_nomel = 'D:/Bases de Imagens/ISIC2020/benign_hair_removal/'

for name in path_isic_maligno:
    d = os.path.join('D:/Bases de Imagens/ISIC2020/malign/', 
                    name)
    data_malign.append(d)


kernel = cv2.getStructuringElement(1,(23,23))

for name in path_isic_benigno:
    d = os.path.join('D:/Bases de Imagens/ISIC2020/benign/', 
                    name)
    data_benign.append(d)
    
i = 0
    
for im in data_malign:
    
    img = cv2.imread(im, cv2.IMREAD_COLOR)
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg= cv2.GaussianBlur(blackhat,(11,11),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,mask,1,cv2.INPAINT_TELEA)
    aux = os.path.split(data_malign[i])
    i = i+1
    path = (out_dir_mel + str(aux[1]))
    cv2.imwrite(path, dst)

i =0

for im in data_benign:
    
    img = cv2.imread(im, cv2.IMREAD_COLOR)
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg= cv2.GaussianBlur(blackhat,(11,11),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,mask,1,cv2.INPAINT_TELEA)
    aux = os.path.split(data_malign[i])
    i = i+1
    path = (out_dir_nomel + str(aux[1]))
    cv2.imwrite(path, dst)

    

    