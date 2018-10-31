'''
Created by Ping Yuhan on Sep. 18, 2018

This is the scratch of the nerual network including build, training, test, tensorboard,etc.

Latest updated on Oct. 11, 2018 
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K 
from keras.callbacks import TensorBoard

height, width = 200, 200 # image dimension
trainingDataDir = 'Data/Training Data'
testDataDir = 'Data/Test Data'
logDir = 'log/'

trainingDataCount = 7260
testDataCount = 240 #total data 7500
epochs = 64
batchsize = 16

#check the dimention order of input image

if K.image_data_format() == 'channel_first':
    inputShape = (3, height, width)
else:
    inputShape = (height, width, 3)

#model scratch
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(24))
model.add(Activation('softmax'))

model.summary() #print the model structure

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

#data augmentation
trainingDA = ImageDataGenerator(rescale = 1. /255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

testDA = ImageDataGenerator(rescale = 1. /255)

trainingGenerator = trainingDA.flow_from_directory(trainingDataDir, target_size = (height, width), 
                                                   batch_size = batchsize, class_mode = 'categorical')
testGenerator = testDA.flow_from_directory(testDataDir, target_size = (height, width), 
                                                   batch_size = batchsize, class_mode = 'categorical')

#call tensorboard to view the training process
tb_cb = TensorBoard(log_dir=logDir, write_graph=True, write_images=True, write_grads = True ,histogram_freq = 0)
cbks = [tb_cb]

model.fit_generator(trainingGenerator, steps_per_epoch = trainingDataCount // batchsize, epochs = epochs, verbose = 1, 
                    validation_data = testGenerator, validation_steps = testDataCount // batchsize, callbacks = cbks)

# score = model.evaluate(testDataDir, testGenerator.classes, verbose = 1)
score = model.evaluate_generator(testGenerator)
print('Test accuracy = ', score[1])
model.save_weights('Model/weights.h5')
model.save('Model/model.h5')