'''
Created by Ping Yuhan on Sep. 18, 2018

This is main program of ASL2TEXT including hand detection, real-time computer vision, model loading and prediction.

Latest updated on Oct. 11, 2018 
'''

from keras.models import load_model
from preprocessing import squarePadding, preProcessForNN
from cv2 import imread

import numpy as np 
import copy
import math
import string
import cv2
import time


# parameter settings for ROI(region of interest)
roi_x = 0.55 
roi_y = 0.9

# parameter settings
threshold = 65 
blur = 9 # gaussian blur
bg_threshold = 50 # background threshold value
learningRate = 0 # learning rate for MOG2

bg_extraction = 0 # backgroun subtracion or not
checkFinger = False # check fingers

# background subtraction
def removeBG(frame):
    backgroundMask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    backgroundMask = cv2.erode(backgroundMask, kernel, iterations=1)
    subtractedFrame = cv2.bitwise_and(frame, frame, mask=backgroundMask)
    isBgExtraction = 1
    return subtractedFrame

# hand detection
def handDetection(contour, cvxWindow):
    # hand detection by calculating the convexy hull and convexity defects
    cvxHull = cv2.convexHull(contour, returnPoints = False)
    if len(cvxHull) > 2:
        cvxDef = cv2.convexityDefects(contour, cvxHull)
        if type(cvxDef) != type(None):
            count = 0
            for i in range(cvxDef.shape[0]):
                start_of_cvxDef, end_of_cvxDef, far_of_cvxDef, depth_of_cvxDef = cvxDef[i][0]
                # the start point, end point and deepest point forms a triangle, if the bottom angle is less than
                # 90 degrees, there are two fingers alongside
                startPoint = tuple(contour[start_of_cvxDef][0])
                endPoint = tuple(contour[end_of_cvxDef][0])
                deepestPoint = tuple(contour[far_of_cvxDef][0])

                v1 = math.sqrt((endPoint[0] - startPoint[0]) ** 2 + (endPoint[1] - startPoint[1]) ** 2)
                v2 = math.sqrt((deepestPoint[0] - startPoint[0]) ** 2 + (deepestPoint[1] - startPoint[1]) ** 2)
                v3 = math.sqrt((endPoint[0] - deepestPoint[0]) ** 2 + (endPoint[1] - deepestPoint[1]) ** 2)

                angle = math.acos((v2 ** 2 + v3 ** 2 - v1 ** 2) / (2 * v2 * v3)) # calculate the consine value
                if angle < math.pi / 2:
                    #if bottom angle < 90, there is a finger alongside
                    count += 1
                    cv2.circle(cvxWindow, deepestPoint, 6, (255, 0, 0), -1) # draw bottom points
            return True, count 
        return False, 0 

# Personalization of which model to use 
modelname = input('Please specify the model to use, vgg16/mobilenet/none for self-built model.')
if modelname == 'vgg16':
    print('You have selected the VGG16 based model.')
    model = load_model('Model/vgg16model.h5')
elif modelname == 'mobilenet':
    print('You have selected the MobileNet based model.')
    model = load_model('Model/mobmodel.h5')
else:
    print('You have selected the self-built model.')
    model = load_model('Model/selfmodel.h5')


# build the alphabet dictionary base on category of numbers
alphabetDict = {pos : alphabet for pos, alphabet in enumerate (string.ascii_uppercase)}

# real-time computer vision
cameraIndex = int(input('Please specify which camera to use, 0 for default webcam: '))
camera = cv2.VideoCapture(cameraIndex)
camera.set(10, 200) # set the brightness
#keyboadr operation commands
print('Press "b" to extract background or press "r" to reset the extraction.')

# start time, for counting FPS
fps = 0
startTime = time.time()

# camera loop
while camera.isOpened():
    success, frame = camera.read()
    fps += 1 # count FPS
    frame = cv2.bilateralFilter(frame, 5, 50, 100) # bilateral filter
    frame = cv2.flip(frame, 1) # mirror the frame
    # ROI
    cv2.rectangle(frame, (int(roi_x * frame.shape[1]), 0), (frame.shape[1], int(roi_y * frame.shape[0])), 
              (255, 0, 0), 2)

    if bg_extraction == 1:
        img = removeBG(frame)
        img = img[0: int(roi_y * frame.shape[0]), int(roi_x * frame.shape[1]):frame.shape[1]]
        cv2.imshow('Mask', img)

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurImg = cv2.GaussianBlur(grayImg, (blur, blur), 0)
        cv2.imshow('Blur', blurImg) # blur image

        # dynamic adjust threshold according to the light condition
        # sample the center&top of the image to get the light condition by its intensity
        w, h = np.shape(img)[: 2]
        lightLv = grayImg[int(h / 100)][int(w / 2)]
        thresLv = bg_threshold + lightLv
        print('The threshold changes to : ', thresLv)

        success, thresImg = cv2.threshold(blurImg, thresLv, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresImg = cv2.dilate(thresImg, None, iterations = 2)
        cv2.imshow('Threshold Image', thresImg) # dynamic threshold image

        newThresImg = copy.deepcopy(thresImg)
        _, cnt, hierarchy = cv2.findContours(newThresImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cvxWindow = np.zeros(img.shape, np.uint8) # define the window for showing convex hull and contour
        maxArea = -1
        length = len(cnt)
        if length > 0:
            for i in range(len(cnt)):
                cur = cnt[i]
                area = cv2.contourArea(cur)
                if area > maxArea:
                    maxArea = area
                    ci = i 
            maxCnt = cnt[ci] # find the largest contour area to get the largest contour which is the contour of hand
            
            cvxHull = cv2.convexHull(maxCnt)
            moment = cv2.moments(maxCnt)
            if moment['m00'] != 0:
                # center coordinates, x = M10/M00, y = M01/M00
                center_x = int(moment['m10'] / moment['m00'])
                center_y = int(moment['m01'] / moment['m00'])
                center = (center_x, center_y)

            cv2.drawContours(cvxWindow, [maxCnt], 0, (0, 255, 0), 2)
            cv2.drawContours(cvxWindow, [cvxHull], -1, (0, 0, 255), 3)

            # if press 'c', call hand detection to check finger numbers
            fingerFlag, fingerCount = handDetection(maxCnt, cvxWindow)
            if checkFinger == True:
                if fingerFlag == True:
                    print('Finger numbers: ' + fingerCount)

        cv2.imshow('Output', cvxWindow)
        img_convert_ndarray = np.array(newThresImg)
        # cv2.imshow('For NN', img_convert_ndarray)
        hand = preProcessForNN(cvxWindow)

        # predictions by model

        predictRst = model.predict(hand, batch_size = None, verbose = 0, steps = 1)
        topRst = np.argmax(predictRst) # the max possibility alphabet
        if np.max(predictRst) >= 0.05:
            result = alphabetDict[topRst]
            # extract top 3 predictions
            predictionRstList = np.argsort(predictRst)[0]
            # 2nd & 3rd predictions
            scdRst = alphabetDict[predictionRstList[-2]]
            thrdRst = alphabetDict[predictionRstList[-3]]
            # text region settings
            textWidth = int(camera.get(3) + 0.25)
            textHeight = int(camera.get(4) + 0.25)

            # show the text
            cv2.putText(frame, text=result,
                        org=(textWidth // 2 + 50, textHeight // 2 + 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=10, color=(255, 255, 0),
                        thickness=15, lineType=cv2.LINE_AA)

            # Annotate image with second most probable prediction (displayed on bottom left)
            cv2.putText(frame, text=scdRst,
                        org=(textWidth // 3 + 100, textHeight // 1 + 5),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=6, color=(0, 255, 0),
                        thickness=6, lineType=cv2.LINE_AA)

            # Annotate image with third probable prediction (displayed on bottom right)
            cv2.putText(frame, text=thrdRst,
                        org=(textWidth // 2 + 120, textHeight // 1 + 5),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=6, color=(0, 0, 255),
                        thickness=6, lineType=cv2.LINE_AA)       
    
    cv2.imshow('Camera', frame)  
              
    # keyboard operation commands
    key = cv2.waitKey(10)
    if key == 27:
        #ESC tp exit
        break
    elif key == ord('b'):
        # start background extraction
        bgModel = cv2.createBackgroundSubtractorMOG2()
        bg_extraction = 1
        print('background extraction performed. You can press "r" to reset.')
    elif key == ord('r'):
        # reset background
        bgModel = None
        bg_extraction = 0
        checkFinger = False
        print('System reset.')
    elif key == ord('c'):
        # finger count
        checkFinger = True
        print('Finger checker is on. ')
    elif key == ord('p'):
        # print the cvxWindow
        h, w = img.shape[: 2]
        printImg = cv2.resize(img, (h, w), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('Data/Output/cvxWindow.jpg', cvxWindow)
        print('cvxWindow has been printed.')


# analysis FPS
endTime = time.time()
FPS = fps / (endTime - startTime)
print('[INFO] Approximate FPS = {:.2f}'.format(FPS))

# release the camera and close window
camera.release()
cv2.destroyAllWindows()
