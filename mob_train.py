'''
Created by Ping Yuhan on Sep. 18, 2018

This is the construction and training of MobileNet based network including transfer learning

Latest updated on Oct. 16, 2018 
'''



from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.applications.mobilenet import MobileNet
from keras import backend as K 
from keras.callbacks import TensorBoard
import numpy as np
import os


height, width = 224, 224 # image dimension

# Dir settings
trainingDataDir = 'Data/Training Data2'
testDataDir = 'Data/Test Data2'
logDir = 'log3/'

# parameter settings
trainingDataCount = 7260
testDataCount = 240 
batch_size = 16
epochs = 64
shape = (224, 224, 3)

# Import original MobileNet
Network = MobileNet
model = Network(include_top=False, weights="imagenet", input_tensor=Input(shape=shape))
print("Original MobileNet has been successfully loaded. ")

# Transfer learninig
# preserve basic model layers
for layer in model.layers:
    layer.trainable = False

# Classification block for transfer learning
block_model = Sequential()
block_model.add(Flatten(input_shape=model.output_shape[1:]))
block_model.add(Dense(256, activation='relu')) # previous outputs -> [0, 255]
block_model.add(Dropout(0.2))
block_model.add(Dense(26, activation='softmax')) # previous outputs -> A to Z
specialMob = Model(inputs=model.input, outputs=block_model(model.output))

#print the model structure
specialMob.summary()

specialMob.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# data augmentation
trainingDA = ImageDataGenerator(rescale=1., featurewise_center=True, rotation_range=15.0, width_shift_range=0.15, height_shift_range=0.15,)

trainingDA.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

train_generator = trainingDA.flow_from_directory(trainingDataDir, target_size=(height, width),
                                                    batch_size=batch_size, class_mode="categorical")

testDA = ImageDataGenerator(rescale=1., featurewise_center=True)

testDA.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

test_generator = testDA.flow_from_directory(testDataDir, target_size=(height, width),
                                                batch_size=batch_size, class_mode="categorical")

# set up tensorboard
tb_cb = TensorBoard(log_dir=logDir, write_graph=True, write_images=True, write_grads = True ,histogram_freq = 0)
cbks = [tb_cb]

specialMob.fit_generator(generator=train_generator, steps_per_epoch=trainingDataCount // batch_size, epochs=epochs,
                         verbose=1, validation_data=test_generator, validation_steps=testDataCount // batch_size,
                         callbacks=cbks)

score = specialMob.evaluate_generator(test_generator)
print('Test accuracy = ', score[1])
specialMob.save_weights('Model/mobweights.h5')
specialMob.save('Model/mobmodel.h5')
