'''
Created by Ping Yuhan on Sep. 18, 2018

This is for data pre-processing like padding, zero centered, etc.

This is also used for live prediction, captured frames needs pre-process for cnn.

Latest updated on Oct. 10, 2018 
'''


import cv2
from glob import glob
import numpy as np 
from PIL import Image



def squarePadding(img, paddingColor = [0, 0, 0]):
	#this is for square padding to reshape the training data or captured frame to be suitable for nn
	#padding color is black
	height = img.shape[0]
	width = img.shape[1]
	padLength = np.abs(height - width) // 2 #the amount of padding

	if height > width:
		padTop = 0
		padBottom = 0
		padLeft = padLength
		padRight = padLength
		padImg = cv2.copyMakeBorder(src = img, top = padTop, bottom = padBottom, 
									left = padLeft, right = padRight, borderType = cv2.BORDER_CONSTANT,
									value = paddingColor)
	elif height < width:
		padTop = padLength
		padBottom = padLength
		padLeft = 0
		padRight = 0
		padImg = cv2.copyMakeBorder(src = img, top = padTop, bottom = padBottom, 
									left = padLeft, right = padRight, borderType = cv2.BORDER_CONSTANT,
									value = paddingColor)
	else:
		padImg = img.copy()

	return padImg

def preProcessForNN(img, size = 200, color = True):
	#this is for reshape and re-scale color range for nn
	img = cv2.resize(img, (size, size))
	img = img.astype(np.float32)/255.0 #normalization
	img = img[:, :, ::-1] #mirror the img from RGB to BGR
	imgForNN = np.expand_dims(img, axis = 0) #expand the img to [1, size, size, 3] for nn input

	return imgForNN


#=====================================code test here=======================
# a = np.array([1, 2, 3])
# print(a, a.shape)
# print(a[0])
# b = np.expand_dims(a, axis = 0)
# print(b, b.shape)
# print(b[0][0])

# rawimg = Image.open('A.jpg')
# img = np.array(rawimg)
# print(img, img.shape)
# arr = preProcessForNN(img)
# print(arr, arr.shape)
# arr = Image.fromarray(np.uint8(arr))
# arr.save('d.jpg')
# print(img)
# print(img.shape[0], img.shape[1])
# img = cv2.resize(img, (200, 200))
# print(img)
# print(img.shape[0], img.shape[1], img.shape)
# #img = img.astype(np.float32)/255.0
# print(img[:, :, ::-1])
# print(img.shape)
# x = np.expand_dims(img, axis = 0)
# print(x, x.shape)
# convertimg = Image.fromarray(np.uint8(img))
# convertimg.save('c.jpg')
# padImg = squarePadding(img)
# convertimg = Image.fromarray(np.uint8(padImg))
# convertimg.show()
# convertimg.save('b.jpg')