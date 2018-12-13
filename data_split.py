'''
Created by Ping Yuhan on Sep. 18, 2018

This is for splitting original dataset into training data and test data

Latest updated on Oct. 10, 2018 
'''

import os
import cv2
from PIL import Image
import numpy as np
from shutil import copy, copyfile, move
from random import sample

def checkAndCopy(src, des, filename, alphabet):
    #check if the directory has already exists, if so ignore the operation, if not, makedir and copy files
    #dir eg. xxx/A/..., xxx/C/...
    if not os.path.exists(des + alphabet):
        os.makedirs(des + alphabet)
    copyfile(src = src + filename, dst = des + alphabet + '/' + filename)


def checkAndMove(src, des, filename, alphabet):
    #duiring split, files needs to be moved to new dir
    #dir eg. xxx/A/..., xxx/C/...
    if not os.path.exists(des + alphabet):
        os.makedirs(des + alphabet)
    move(src = src, dst = des + alphabet + '/' + filename)


if __name__ == '__main__':
    #default param settings
    folderList = os.listdir('Data/Original/')
    aslAlphabet = ['A', 'B', 'C', 'D', 'E',
                   'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O',
                   'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']

    # resize data for VGG16 and MobileNet
    for file in sorted(folderList):
        img = cv2.imread(filename = 'Data/Original/' + file)
        # images need to be resized to 224 * 224
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
        img = Image.fromarray(img) # convert back to image format
        img.save('Data/VGGandMobile/' + file)
    print('224 resized')
    

    #create training data folder for VGG16 and MobileNet
    folderList2 = os.listdir('Data/VGGandMobile/')
    for file in sorted(folderList2):
        for i in range(len(aslAlphabet)):
            if file[0] == aslAlphabet[i]:
                checkAndCopy(src = 'Data/VGGandMobile/', des = 'Data/Training Data2/', filename = file, alphabet = aslAlphabet[i])
                print('Training data2 of letter ' + aslAlphabet[i] + ' created.')
   
    print('Training data2 creation succeed')

    trainFolder = os.listdir('Data/Training Data2/')
    for folder in sorted(trainFolder):
        files = os.listdir('Data/Training Data2/' + folder)
        samples = sample(files, 10) #randomly select 10 files from each folder
        for samplefile in samples:
            checkAndMove(src = 'Data/Training Data2/' + folder + '/' + samplefile, des = 'Data/Test Data2/', filename = samplefile, alphabet = folder)
        print('Test data of letter ' + folder[0] + ' created.')
    print('Test data2 creation succeed')

    #create training data folder for special designed network
    for file in sorted(folderList):
        for i in range(len(aslAlphabet)):
            if file[0] == aslAlphabet[i]:
                checkAndCopy(src = 'Data/Original/', des = 'Data/Training Data/', filename = file, alphabet = aslAlphabet[i])
                print('Training data of letter ' + aslAlphabet[i] + ' created.')

    print('Training data creation succeed')

    #randomly select files in training data and create test data
    trainFolder = os.listdir('Data/Training Data/')
    for folder in sorted(trainFolder):
        files = os.listdir('Data/Training Data/' + folder)
        samples = sample(files, 10) #randomly select 10 files from each folder
        for samplefile in samples:
            checkAndMove(src = 'Data/Training Data/' + folder + '/' + samplefile, des = 'Data/Test Data/', filename = samplefile, alphabet = folder)
        print('Test data of letter ' + folder[0] + ' created.')
    print('Test data creation succeed')