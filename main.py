import torch
import tensorflow as tf
from tensorflow import keras
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
from keras.applications import MobileNet
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
import skimage
from keras.applications import VGG16

variables_for_classification=2

mobile = tf.keras.applications.mobilenet.MobileNet()
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
base_model.trainable = False
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
preds=Dense(variables_for_classification, activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)


for layer in model.layers:
    layer.trainable=True

train_datagen=ImageDataGenerator('C:/Users/isusu/PycharmProjects/tf2.1/data_with_blur/train',
width_shift_range=0.25, height_shift_range=0.25,
                            horizontal_flip=True,
                            rotation_range=45,
                            brightness_range=[0.5,1.5],
                            zoom_range=[0.5,1.5],
                                 shear_range = 0.2,
                            rescale=1.0/255.0,
                            samplewise_center=True,
                            samplewise_std_normalization=True,

                            )

    #included in our dependencies

val_datagen=ImageDataGenerator('C:/Users/isusu/PycharmProjects/tf2.1/data_with_blur/validation',
                               rescale=1.0 / 255.0,
                               samplewise_center=True,
                               samplewise_std_normalization=True,
                               )

test_datagen=ImageDataGenerator('C:/Users/isusu/PycharmProjects/tf2.1/data_with_blur/test',
                                rescale=1.0 / 255.0,
                                samplewise_center=True,
                                samplewise_std_normalization=True,

                                )


train_iterator=train_datagen.flow_from_directory('C:/Users/isusu/PycharmProjects/tf2.1/data_with_blur/train',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=90,
                                                     shuffle=True,
                                                    seed=42,
                                               )


validation_iterator=val_datagen.flow_from_directory('C:/Users/isusu/PycharmProjects/tf2.1/data_with_blur/validation',
                                              target_size=(224, 224),
                                              color_mode='rgb',
                                              batch_size=38,
                                              shuffle=True,

                                              seed=42)

test_iterator=test_datagen.flow_from_directory('C:/Users/isusu/PycharmProjects/tf2.1/data_with_blur/test',
                                               target_size=(224, 224),
                                               color_mode='rgb',
                                               batch_size=1,
                                               shuffle=False,
                                                seed=42,

                                               )



step_size_train=train_iterator.n//train_iterator.batch_size


early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
save_model = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

opt = tf.keras.optimizers.Adagrad(learning_rate=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

model.fit_generator(train_iterator,
                       steps_per_epoch=step_size_train,
                        epochs=500,
                        validation_data = validation_iterator,
                        validation_steps=4,
                        verbose=1,
                    callbacks=[early]
                        )

loss = model.evaluate_generator(test_iterator, verbose=0)

print("Tested Loss:", loss)

classes = train_iterator.class_indices
print(classes)
accuracy = model.evaluate_generator(test_iterator, steps=4, workers=0)
prediction = model.predict_generator(test_iterator, steps=156, workers=0)
filenames = test_iterator.filenames
print(filenames)
print(prediction)
print(accuracy[0])
print("Accuracy = ", accuracy[1])




