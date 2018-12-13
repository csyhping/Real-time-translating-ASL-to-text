'''
Created by Ping Yuhan on Sep. 18, 2018

This is a test program for testing data augmentation.

Latest updated on Oct. 16, 2018 
'''

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import glob
import os

path = 'F:\\课件/HKU/Dissertation/Code/img/'
gen_path = './img/gen/'

namelist = glob.glob(path + '*/*')
print(namelist)

datagen = ImageDataGenerator(rotation_range = 30)
gen_data = datagen.flow_from_directory(path, batch_size = 2, shuffle = False, save_to_dir = gen_path, 
	save_prefix = 'gen', target_size = (200, 200))

for i in range(3):
	gen_data.next()