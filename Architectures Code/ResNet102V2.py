import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import time, datetime

# Parâmetros e diretórios 
 
train_data_dir = 'D:\\Arquivos_Doutorado\\Codigo-Fonte\\HAM10000-seg-croped\\train'
validation_data_dir = 'D:\\Arquivos_Doutorado\\Codigo-Fonte\\HAM10000-seg-croped\\test'
log_dir = 'D:\\Arquivos_Doutorado\\Codigo-Fonte\\logs'
nb_train_samples = 8015
img_width = 224
img_height = 224
nb_validation_samples = 2100
epochs = 1000
batch_size = 32

# Métricas utilzidas 

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

# Verificação das imagens de entrada 

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# Criação do modelo ResNet101V2

model = ResNet101V2(include_top=True, weights=None, input_tensor=None,
    input_shape=None, pooling=None, classes=7, classifier_activation='softmax')

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=METRICS)

#Aumento de dados Conjunto de treino 

generate_train = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)


generate_test = ImageDataGenerator(rescale =1./255)

# Criação dos conjuntos a partir dos diretórios 

data_train = generate_train.flow_from_directory(train_data_dir,
                                                           target_size = (224, 224),
                                                           batch_size = 32,
                                                           class_mode = 'categorical')
data_test = generate_test.flow_from_directory(validation_data_dir,
                                  target_size = (224, 224),
                                  batch_size = 32,
                                  class_mode = 'categorical')

# Parâmetros para criação de backups

checkpoint_path = "backup\\cp-{epoch:04d}.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(filepath=checkpoint_dir, save_weights_only=True, 
                              monitor='val_accuracy', mode='max', verbose=1)

latest = tf.train.latest_checkpoint(checkpoint_dir)

#model.load_weights(latest) 

# Monitorar as épocas

callback_epochs = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

# Acompanhamento visual 

tensorboard = tf.keras.callbacks.TensorBoard( log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False,
                               update_freq='epoch', profile_batch=2,embeddings_freq=0, embeddings_metadata=None)

# Treinamento do modelo 

model.fit(
    data_train,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=data_test,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[cp_callback, callback_epochs, tensorboard])

# Save weigths ResNet102V2 Training

model.save_weights('ResNet102V2.h5')

#Save arq

ResNet50_json = model.to_json()

with open('ResNet102V2.json','w') as json_file:
    json_file.write(ResNet50_json)
