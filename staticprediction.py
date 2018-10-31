'''
Created by Ping Yuhan on Sep. 18, 2018

This is prediction with static images. A pre-trained model is loaded.

Latest updated on Oct. 12, 2018 
'''

from cv2 import imread
from keras.models import load_model
from PIL import Image 
import numpy as np 
from preprocessing import preProcessForNN, squarePadding

def modelLoad(preTrainedModel):
	#load a pre-trained model
	model = load_model(preTrainedModel)
	return model

def predict(preTrainedModel, img):
	#load a test image, preprocess it to be suiltable as model's input
	testImg = Image.open(img)
	# testImg = squarePadding(testImg)
	# testImg = preProcessForNN(testImg)
	testImg = testImg.resize((200, 200), resample = 0)
	testImg.save('Data/StaticTest/resizedimg.jpg') #Data/StaticTest/
	testImg = imread('Data/StaticTest/resizedimg.jpg')
	testImg = testImg.astype(np.float32) / 255.0
	testImg = testImg[:, :, ::-1] #mirror BGR to RGB

	prediction = preTrainedModel.predict(np.expand_dims(testImg, axis = 0))[0]
	return prediction

def correspondAlphabet(predictionResult, alphabetDict):
	#find the biggest value in prediction result and locate to corresponding alphabet
	maxofPredict = predictionResult.argmax(axis = 0) #locate the max value 

	corrAlphabet = [ch for ch, val in alphabetDict.items() if val == maxofPredict]
	return corrAlphabet[0]

if __name__ == '__main__':
	model = modelLoad('Model/model.h5')
	prediction = predict(preTrainedModel = model, img = 'Data/StaticTest/5.jpg')
	alphabetDict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 
					'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
					'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'S': 17,
					'T': 18, 'U': 19, 'V': 20, 'W': 21, 'X': 22, 'Y': 23} #MISSING LETTER R AND Z
	corrAlphabet = correspondAlphabet(predictionResult = prediction, alphabetDict = alphabetDict)
	print('The prediction result is letter: ', corrAlphabet)

