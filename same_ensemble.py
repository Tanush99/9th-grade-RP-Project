import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNet
import numpy as np
from numpy import array
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adagrad


n_members = 7
models = list()

no_aug_train_datagen=ImageDataGenerator('./data_with_seg/train',
                                    rescale=1.0/255.0,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    )

no_aug_train_iterator = no_aug_train_datagen.flow_from_directory('./data_with_seg/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,)
train_datagen=ImageDataGenerator('./data_with_seg/train',
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

test_datagen=ImageDataGenerator('./data_with_seg/test',
                                            rescale=1.0/255.0,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True)

val_datagen=ImageDataGenerator('./data_with_seg/validation',
                                            rescale=1.0/255.0,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True)




train_iterator=train_datagen.flow_from_directory('./data_with_seg/train',
                                                             target_size=(224,224),
                                                             color_mode='rgb',
                                                             batch_size=90,
                                                             class_mode='categorical',
                                                             shuffle=True,
                                                            seed=42,
)


test_iterator=test_datagen.flow_from_directory('./data_with_seg/test',
                                                      target_size=(224, 224),
                                                      color_mode='rgb',
                                                      batch_size=1,
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                      seed=42)

val_iterator=val_datagen.flow_from_directory('./data_with_seg/validation',
                                                      target_size=(224, 224),
                                                      color_mode='rgb',
                                                      batch_size=38,
                                                      class_mode='categorical',
                                                      shuffle=False,
                                                      seed=42)
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

                    model=Model(inputs=base_model.input,outputs=preds)


                    for layer in model.layers:
                        layer.trainable=True

                    opt = tf.keras.optimizers.Adagrad(lr=0.001)
                    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
                    step_size_train=train_iterator.n//train_iterator.batch_size
                    early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
                    model.fit_generator(train_iterator,
                                           steps_per_epoch=step_size_train,
                                            epochs=175,
                                            verbose=1,
                                            callbacks=[early])

                    #loss = model.evaluate_generator(train_iterator, steps=1)

                    models.append(model)

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
                    x = Dropout(rate=0.2)(x)
                    preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

                    model = Model(inputs=base_model.input, outputs=preds)

                    for layer in model.layers:
                        layer.trainable = True

                    opt = tf.keras.optimizers.Adagrad(lr=0.001)
                    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
                    step_size_train = train_iterator.n // train_iterator.batch_size
                    early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
                    model.fit_generator(train_iterator,
                                        steps_per_epoch=step_size_train,
                                        epochs=175,
                                        verbose=1,
                                        callbacks=[early])

                    # loss = model.evaluate_generator(train_iterator, steps=1)

                    models.append(model)

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

                    model = Model(inputs=base_model.input, outputs=preds)

                    for layer in model.layers:
                        layer.trainable = True

                    opt = tf.keras.optimizers.Adagrad(lr=0.001)
                    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
                    step_size_train = train_iterator.n // train_iterator.batch_size
                    early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)
                    model.fit_generator(train_iterator,
                                        steps_per_epoch=step_size_train,
                                        epochs=175,
                                        verbose=1,
                                        callbacks=[early])

                    # loss = model.evaluate_generator(train_iterator, steps=1)

                    models.append(model)

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

                    model = Model(inputs=base_model.input, outputs=preds)

                    for layer in model.layers:
                        layer.trainable = True

                    opt = tf.keras.optimizers.Adagrad(lr=0.001)
                    model.compile(optimizer=opt, loss="squared_hinge", metrics=["accuracy"])
                    step_size_train = train_iterator.n // train_iterator.batch_size
                    early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
                    model.fit_generator(train_iterator,
                                        steps_per_epoch=step_size_train,
                                        epochs=175,
                                        verbose=1,
                                        callbacks=[early])

                    # loss = model.evaluate_generator(train_iterator, steps=1)

                    models.append(model)

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

                    model = Model(inputs=base_model.input, outputs=preds)

                    for layer in model.layers:
                        layer.trainable = True

                    opt = tf.keras.optimizers.Adagrad(lr=0.001)
                    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
                    step_size_train = no_aug_train_iterator.n // no_aug_train_iterator.batch_size
                    early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
                    model.fit_generator(no_aug_train_iterator,
                                        steps_per_epoch=step_size_train,
                                        epochs=175,
                                        verbose=1,
                                        callbacks=[early])

                    # loss = model.evaluate_generator(train_iterator, steps=1)

                    models.append(model)


def ensemble_predictions(models):
     classes = test_iterator.class_indices
     print(classes)
     filenames = test_iterator.filenames
     print(filenames)
     yhats = [model.predict(test_iterator, steps=156, workers=0) for model in models]
     yhats = array(yhats)
     summed = np.sum(yhats, axis=0)
     outcomes = np.argmax(summed, axis=1)
     print(outcomes)
train()
ensemble_predictions(models)
