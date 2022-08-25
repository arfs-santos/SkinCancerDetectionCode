import numpy as np
import shutil
import os 
import pandas as pd 
from pathlib import PurePath
import cv2

csv_file = pd.read_csv('D:/Bases de Imagens/PAD-UFES/metadata.csv')

images_names = csv_file['img_id']

images_class = csv_file['diagnostic']

#metadata =  pd.read_csv('D:/Bases de Imagens/Classe.csv',index_col=(0))

#images_id = image_names.image

#dummies = pd.get_dummies(metadata).astype('category')

#aux = dummies.idxmax(axis=1)

#classes = np.stack(aux)