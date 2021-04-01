import torch
import tensorflow as tf
import keras
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D, Conv2D, Dropout
from keras.applications import VGG16
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import cv2
import numpy as np
import tensorflowjs as tfjs
import os
import tempfile
from tensorflow import keras
from tensorflow import lite

mobile = tf.keras.applications.VGG16()
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.VGG16.preprocess_input(img_array_expanded_dims)


base_model=VGG16(include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model.trainable = False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dense(1024,activation='relu')(x)#we add dense layers so that the model can learn more complex functions and classify for better results.#dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)


for layer in model.layers:
    layer.trainable=True

train_datagen=ImageDataGenerator('C:/Users/isusu/PycharmProjects/tf2.1/data_with_seg/train',
                               width_shift_range=0.25, height_shift_range=0.25,
                               horizontal_flip=True,
                               rotation_range=45,
                            brightness_range=[0.5,1.5],
                                zoom_range=[0.5,1.5],
                            rescale=1.0/255.0,
                            samplewise_center=True,
                            samplewise_std_normalization=True,
                            shear_range=0.2
                               )

    #included in our dependencies

val_datagen=ImageDataGenerator('C:/Users/isusu/PycharmProjects/tf2.1/data_with_seg/validation',
                                    rescale=1.0/255.0,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)


train_iterator=train_datagen.flow_from_directory('C:/Users/isusu/PycharmProjects/tf2.1/data_with_seg/train',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=90,
                                                     class_mode='categorical',
                                                     shuffle=True,
                                                    seed=42,


                                               )


validation_iterator=val_datagen.flow_from_directory('C:/Users/isusu/PycharmProjects/tf2.1/data_with_seg/validation',
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              batch_size=38,
                                              class_mode='categorical',
                                              shuffle=True,
                                              seed=42)


step_size_train=train_iterator.n//train_iterator.batch_size

early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
save_model = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit_generator(train_iterator,
                       steps_per_epoch=step_size_train,
                        epochs=500,
                        validation_data = validation_iterator,
                        validation_steps=2,
                        verbose=1,
                        callbacks=[early]
                        )

loss = model.evaluate_generator(train_iterator, steps=1)

print("Tested Loss:", loss)


















