import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from time import datetime, time
from tensorflow.python.keras.callbacks import TensorBoard

train_data_dir = 'D:\Alan-Santos\Codigo-Fonte\HAM10000-seg-croped\train'
validation_data_dir = 'D:\Alan-Santos\Codigo-Fonte\HAM10000-seg-croped\test'
checkpoint_path = 'D:\Alan-Santos\Codigo-Fonte\backup\cp1.ckpt'
nb_train_samples = 10015
img_width = 224
img_height = 224
nb_validation_samples = 10015
epochs = 1000
batch_size = 32
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = InceptionV3(include_top=True, weights=None, input_tensor=None, 
                    input_shape=input_shape, pooling=None, classes=7, classifier_activation='softmax')

model.load_weights(checkpoint_path)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=METRICS)


generate_train = ImageDataGenerator(rescale=1. / 255,
                                    rotation_range=7,
                                    horizontal_flip=True,
                                    shear_range=0.2,
                                    height_shift_range=0.07,
                                    zoom_range=0.2)

generate_test = ImageDataGenerator(rescale=1. / 255)


data_train = generate_train.flow_from_directory(train_data_dir,
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')

data_test = generate_test.flow_from_directory(validation_data_dir,
                                              target_size=(227, 227),
                                              batch_size=32,
                                              class_mode='categorical')


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callback_epochs = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

tensorboard = TensorBoard(log_dir ="/logs/{}".format(time()))

model.fit(
    data_train,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=data_test,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[cp_callback, callback_epochs, tensorboard])

model.save_weights('InceptionV3.h5')


inceptionV3_json = model.to_json()

with open('inceptionV3.json', 'w') as json_file:
    json_file.write(inceptionV3_json)

print('Arquivo JSON criado...')