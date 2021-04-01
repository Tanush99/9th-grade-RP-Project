import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNet
import numpy as np


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
x = Dropout(rate=0.2)(x)
preds=Dense(variables_for_classification, activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)


for layer in model.layers:
    layer.trainable=True

train_datagen5=ImageDataGenerator('./cv 5/train',
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


test_datagen5=ImageDataGenerator('./cv 5/test',
                                rescale=1.0 / 255.0,
                                samplewise_center=True,
                                samplewise_std_normalization=True,

                                )


train_iterator5=train_datagen5.flow_from_directory('./cv 5/train',
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=90,
                                                     shuffle=True,
                                                    seed=42,



                                               )

test_iterator5=test_datagen5.flow_from_directory('./cv 5/test',
                                               target_size=(224, 224),
                                               color_mode='rgb',
                                               batch_size=39,
                                               shuffle=False,
                                                seed=42,

                                               )



step_size_train=train_iterator5.n//train_iterator5.batch_size


early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
save_model = ModelCheckpoint('best_model.h5', monitor='loss', mode='min', save_best_only=True, verbose=1)

opt = tf.keras.optimizers.Adagrad(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

model.fit_generator(train_iterator5,
                       steps_per_epoch=step_size_train,
                        epochs=500,
                        verbose=1,
                    callbacks=[early]
                        )

loss = model.evaluate_generator(test_iterator5, verbose=0)

print("Tested Loss:", loss)

classes = train_iterator5.class_indices
print(classes)
prediction = model.evaluate_generator(test_iterator5, steps=4, workers=0)
filenames = test_iterator5.filenames
print(filenames)
print(prediction[0])
print("Accuracy = ", prediction[1])




