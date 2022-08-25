import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  os
from PIL import Image

train_data_dir = 'D:\Alan-Santos\Base de Imagens\Train_HAM10000'
validation_data_dir = 'D:\Alan-Santos\Base de Imagens\Test_Other'
nb_train_samples = 7923
img_width = 224
img_height = 224
nb_validation_samples = 1677
epochs = 1000
batch_size = 32


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



model = tf.keras.applications.vgg16.VGG16(include_top=True, weights=None, 
                                          input_tensor=None, input_shape=input_shape, 
                                          pooling=None, classes=1)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

generate_train = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

generate_test = ImageDataGenerator(rescale =1./255)


data_train = generate_train.flow_from_directory(train_data_dir,
                                                           target_size = (224, 224),
                                                           batch_size = 1,
                                                           class_mode = 'binary')
data_test = generate_test.flow_from_directory(validation_data_dir,
                                  target_size = (224, 224),
                                  batch_size = 1,
                                  class_mode = 'binary')


checkpoint_path = 'D:\\backup'

checkpoint_dir = os.path.dirname(checkpoint_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


callback_epochs = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)



model.fit(data_train, steps_per_epoch = nb_train_samples, epochs=epochs,
                    validation_data=data_test, validation_steps= nb_validation_samples, callbacks=[cp_callback, callback_epochs])


#save weigths AlexNet Training

model.save_weights('VGG16.h5')

VGG16_json = model.to_json()

with open('VGG16.json','w') as json_file:
    json_file.write(VGG16_json)