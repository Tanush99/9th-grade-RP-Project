import tensorflow as tf
import keras
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Conv2D, Dropout
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np

models = list()
opt = tf.keras.optimizers.Adagrad(lr=0.001)
train1_datagen=ImageDataGenerator('./cv 1/train',
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


train1_iterator=train1_datagen.flow_from_directory('./cv 1/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train2_datagen=ImageDataGenerator('./cv 2/train',
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


train2_iterator=train2_datagen.flow_from_directory('./cv 2/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train3_datagen=ImageDataGenerator('./cv 3/train',
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


train3_iterator=train3_datagen.flow_from_directory('./cv 3/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train4_datagen=ImageDataGenerator('./cv 4/train',
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


train4_iterator=train4_datagen.flow_from_directory('./cv 4/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train5_datagen=ImageDataGenerator('./cv 5/train',
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


train5_iterator=train5_datagen.flow_from_directory('./cv 5/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)


train6_datagen=ImageDataGenerator('./cv 6/train',
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


train6_iterator=train6_datagen.flow_from_directory('./cv 6/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train7_datagen=ImageDataGenerator('./cv 7/train',
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


train7_iterator=train7_datagen.flow_from_directory('./cv 7/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)


train8_datagen=ImageDataGenerator('./cv 8/train',
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


train8_iterator=train8_datagen.flow_from_directory('./cv 8/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train9_datagen=ImageDataGenerator('./cv 9/train',
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


train9_iterator=train9_datagen.flow_from_directory('./cv 9/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)

train10_datagen=ImageDataGenerator('./cv 10/train',
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


train10_iterator=train10_datagen.flow_from_directory('./cv 10/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)



def train():
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
            x=Dropout(rate=0.2)(x)
            preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

            model1=Model(inputs=base_model.input,outputs=preds)

            for layer in model1.layers:
                layer.trainable=True

            model1.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train=train1_iterator.n//train1_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model1.fit_generator(train1_iterator,
                                   steps_per_epoch=step_size_train,
                                    epochs=50,
                                    verbose=1,
                                    callbacks=[early]
                                    )

            models.append(model1)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model2 = Model(inputs=base_model.input, outputs=preds)

            for layer in model2.layers:
                layer.trainable = True

            model2.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train2_iterator.n // train2_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model2.fit_generator(train2_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model2)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model3 = Model(inputs=base_model.input, outputs=preds)

            for layer in model3.layers:
                layer.trainable = True

            model3.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train3_iterator.n // train3_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model3.fit_generator(train3_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model3)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model4 = Model(inputs=base_model.input, outputs=preds)

            for layer in model4.layers:
                layer.trainable = True

            model4.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train4_iterator.n // train4_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model4.fit_generator(train4_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model4)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model5 = Model(inputs=base_model.input, outputs=preds)

            for layer in model5.layers:
                layer.trainable = True

            model5.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train5_iterator.n // train5_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model5.fit_generator(train5_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model5)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model6 = Model(inputs=base_model.input, outputs=preds)

            for layer in model6.layers:
                layer.trainable = True

            model6.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train6_iterator.n // train6_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model6.fit_generator(train6_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model6)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model7 = Model(inputs=base_model.input, outputs=preds)

            for layer in model7.layers:
                layer.trainable = True

            model7.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train7_iterator.n // train7_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model7.fit_generator(train7_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model7)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model8 = Model(inputs=base_model.input, outputs=preds)

            for layer in model8.layers:
                layer.trainable = True

            model8.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train8_iterator.n // train8_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model8.fit_generator(train8_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model8)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model9 = Model(inputs=base_model.input, outputs=preds)

            for layer in model9.layers:
                layer.trainable = True

            model9.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train9_iterator.n // train9_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model9.fit_generator(train9_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model9)

            mobile = tf.keras.applications.mobilenet.MobileNet()

            def prepare_image(file):
                img_path = ''
                img = image.load_img(img_path + file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

            base_model = MobileNet(weights='imagenet',
                                   include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
            base_model.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

            model10 = Model(inputs=base_model.input, outputs=preds)

            for layer in model10.layers:
                layer.trainable = True

            model10.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

            step_size_train = train10_iterator.n // train10_iterator.batch_size
            early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
            model10.fit_generator(train10_iterator,
                                 steps_per_epoch=step_size_train,
                                 epochs=50,
                                 verbose=1,
                                 callbacks=[early]
                                 )

            models.append(model10)


train()

