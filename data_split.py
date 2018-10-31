'''
Created by Ping Yuhan on Sep. 18, 2018

This is for splitting original dataset into training data and test data

Latest updated on Oct. 10, 2018 
'''

import os
from shutil import copyfile, move
from random import sample

def copyFile(src, des, filename, alphabet):
	#check if the directory has already exists, if so ignore the operation, if not, makedir and copy files
	#dir eg. xxx/A/..., xxx/C/...
	if not os.path.exists(des + alphabet):
		os.makedirs(des + alphabet)
	copyfile(src = src + filename, dst = des + alphabet + '/' + filename)


def moveFile(src, des, filename, alphabet):
	#duiring split, files needs to be moved to new dir
	#dir like copyFile()
	if not os.path.exists(des + alphabet):
		os.makedirs(des + alphabet)
	move(src = src, dst = des + alphabet + '/' + filename)


if __name__ == '__main__':
	#default param settings
	folderList = os.listdir('Data/RawImages/')
	aslAlphabet = ['A', 'B', 'C', 'D', 'E',
				   'F', 'G', 'H', 'I', 'J',
				   'K', 'L', 'M', 'N', 'O',
				   'P', 'Q', 'R', 'S', 'T',
				   'U', 'V', 'W', 'X', 'Y', 'Z']
	#create training data folder
	for file in sorted(folderList):
		for i in range(len(aslAlphabet)):
			if file[0] == aslAlphabet[i]:
				copyFile(src = 'Data/RawImages/', des = 'Data/Training Data/', filename = file, alphabet = aslAlphabet[i])
				print('Training data of letter ' + aslAlphabet[i] + ' created.')

	print('Training data creation succeed')

	#randomly select files in training data and create test data
	trainFolder = os.listdir('Data/Training Data/')
	for folder in sorted(trainFolder):
		files = os.listdir('Data/Training Data/' + folder)
		samples = sample(files, 10) #randomly select 10 files from each folder
		for samplefile in samples:
			moveFile(src = 'Data/Training Data/' + folder + '/' + samplefile, des = 'Data/Test Data/', filename = samplefile, alphabet = folder)
		print('Test data of letter ' + folder[0] + ' created.')
	print('Test data creation succeed')