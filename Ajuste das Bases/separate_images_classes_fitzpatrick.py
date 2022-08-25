import numpy as np
import shutil
import os 
import pandas as pd 
from pathlib import PurePath
import cv2
import csv

path_class = 'D:/Bases de Imagens/Fitzpatrick17k/images_class.csv'

path_names = 'D:/Bases de Imagens/Fitzpatrick17k/images_names.csv'

images_class = pd.read_csv(path_class)

images_class = images_class.three_partition_label

images_names = pd.read_csv(path_names)

images_names = images_names.md5hash

data = []

for name in images_names:
    d = os.path.join('D:/Bases de Imagens/Fitzpatrick17k/images/', 
                    name +'.jpg')
    data.append(d)

path_malign = 'D:/Bases de Imagens/Fitzpatrick17k/malign'

path_benign = 'D:/Bases de Imagens/Fitzpatrick17k/benign'  

path_non_neoplastic = 'D:/Bases de Imagens/Fitzpatrick17k/non-neoplastic' 

it = 0

for cl in images_class:
    if cl == 'malignant':
        nome = PurePath(data[it]).name
        dest = os.path.join(path_malign, nome)
        shutil.copy(data[it], dest)
        it= it+1
        continue   
    
    elif cl =='benign':
       nome = PurePath(data[it]).name
       dest = os.path.join(path_benign, nome)
       shutil.copy(data[it], dest)
       it= it+1
       continue
   
    elif cl =='non-neoplastic':
      nome = PurePath(data[it]).name
      dest = os.path.join(path_non_neoplastic, nome)
      shutil.copy(data[it], dest)
      it= it+1
      continue
   
    else:
       break
   
