import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os


train_data_dir = 'D:/Bases de Imagens'
validation_data_dir = 'D:/Bases de Imagens'
nb_train_samples = 25331
img_width = 224
img_height = 224
nb_validation_samples = 25331
epochs = 1000
batch_size = 32



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = ResNet50(include_top=True, weights=None, 
                 input_tensor=None, input_shape=input_shape, 
                 pooling=None, classes=8, classifier_activation='softmax')

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics='categorical_accuracy')

generate_train = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

generate_test = ImageDataGenerator(rescale =1./255)


data_train = generate_train.flow_from_directory(train_data_dir,
                                                           target_size = (224, 224),
                                                           batch_size = 32,
                                                           class_mode = 'categorical')
data_test = generate_test.flow_from_directory(validation_data_dir,
                                  target_size = (224, 224),
                                  batch_size = 32,
                                  class_mode = 'categorical')


checkpoint_path = "backup/cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, 
                              monitor='val_accuracy', mode='max', save_best_only=True)

latest = tf.train.latest_checkpoint(checkpoint_dir)

#model.load_weights(latest)

model.fit(
    data_train,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=data_test,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[cp_callback])

#save weigths AlexNet Training

model.save_weights('ResNet50.h5')

ResNet50_json = model.to_json()

with open('ResNet50.json','w') as json_file:
    json_file.write(ResNet50_json)