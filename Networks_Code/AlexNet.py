import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import  os

# Methrics

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc')
]

# AlexNet Model

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(227, 227, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
model.add(Activation('relu'))

# Max Pooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile AlexNet

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

# Data Augmentation 

generate_train = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

generate_test = ImageDataGenerator(rescale =1./255)

# Directory Images Train and Test

data_train = generate_train.flow_from_directory('D:/Google Drive/Projeto_Doutorado (1)/Codigo-Fonte/Base de Imagens/HAM10000/HA1000-bin/train',
                                                           target_size = (227, 227),
                                                           batch_size = 32,
                                                           class_mode = 'binary',
                                                           color_mode="rgb"
                                                           )
data_test = generate_test.flow_from_directory('D:/Google Drive/Projeto_Doutorado (1)/Codigo-Fonte/Base de Imagens/HAM10000/HA1000-bin/test',
                                  target_size = (227, 227),
                                  batch_size = 32,
                                  class_mode = 'binary',
                                  color_mode="rgb")




# checkpoint save

checkpoint_path = 'D:/Google Drive/Projeto_Doutorado (1)/Codigo-Fonte/Arquivos Redes Treinadas/AlexNet-Binary/backup'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Model.load_weights(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor='precision',
                                                 mode='max',
                                                 verbose=1)
# Train parameters

model.fit(data_train, steps_per_epoch = 8015/32, epochs=100, 
                    validation_data=data_test, validation_steps=2100/32, callbacks=[cp_callback])


#save weigths AlexNet Training

model.save_weights('D:/Google Drive/Projeto_Doutorado (1)/Codigo-Fonte/Arquivos Redes Treinadas/AlexNet-Binary/AlexNet.h5')


#save model AlexNet Training

# AlexNet_json = model.to_json()

# with open('AlexNet.json','w') as json_file:
#     json_file.write(AlexNet_json)



