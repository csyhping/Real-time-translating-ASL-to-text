'''
Created by Ping Yuhan on Sep. 18, 2018

This is the construction and training of VGG16 based network including transfer learning

Latest updated on Oct. 16, 2018 
'''


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.applications.vgg16 import VGG16
from keras import backend as K 
from keras.callbacks import TensorBoard
import numpy as np
import os

# parameter settings
height, width = 224, 224 # image dimension
batch_size = 16
epochs = 64
shape = (224, 224, 3)
trainingDataDir = 'Data/Training Data2'
testDataDir = 'Data/Test Data2'
logDir = 'log3/'
trainingDataCount = 7260
testDataCount = 240 #total data 7500


# load model
print("[INFO] loading {}...".format('vgg16'))
Network = VGG16
model = Network(include_top=False,
                weights="imagenet",
                input_tensor=Input(shape=shape))
print("[INFO] model loaded.")


# freeze base model layers
for layer in model.layers:
    layer.trainable = False


# Classification block
print("[INFO] creating classification block")
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(26, activation='softmax'))

# Join model + classification block
specialVGG = Model(inputs=model.input,
                 outputs=top_model(model.output))

#print the model structure
specialVGG.summary()

# compile model
print("[INFO] compiling model")
specialVGG.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# data augmentation
trainingDA = ImageDataGenerator(rescale=1., featurewise_center=True, rotation_range=15.0, width_shift_range=0.15, height_shift_range=0.15,)

trainingDA.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

train_generator = trainingDA.flow_from_directory(trainingDataDir, target_size=(height, width),
                                                    batch_size=batch_size, class_mode="categorical")

# define validation data generator
testDA = ImageDataGenerator(rescale=1., featurewise_center=True)

testDA.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

test_generator = testDA.flow_from_directory(testDataDir, target_size=(height, width),
                                                batch_size=batch_size, class_mode="categorical")

# set up tensorboard
tb_cb = TensorBoard(log_dir=logDir, write_graph=True, write_images=True, write_grads = True ,histogram_freq = 0)
cbks = [tb_cb]

# train model
specialVGG.fit_generator(generator=train_generator, steps_per_epoch=trainingDataCount // batch_size, epochs=epochs,
                         verbose=1, validation_data=test_generator, validation_steps=testDataCount // batch_size,
                         callbacks=cbks)


score = specialVGG.evaluate_generator(test_generator)
print('Test accuracy = ', score[1])
specialVGG.save_weights('Model/vgg16weights.h5')
specialVGG.save('Model/vgg16model.h5')
