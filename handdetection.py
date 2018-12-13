'''
Created by Ping Yuhan on Sep. 18, 2018

This is a test program for testing camera control and hand detection techniques.

Latest updated on Oct. 16, 2018 
'''

import cv2
import numpy as np 
import copy
import math

# parameter settings for ROI(region of interest)
roi_x = 0.5
roi_y = 0.8
threshold = 60 # threshold value
blur = 5 # Gaussian blur value
bg_threshold = 50 # background threshold value
learningRate = 0 # learning rate for MOG2

bg_extraction = 0 #backgroun subtraction or not
checkFinger = False # check fingers

# background subtraction
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    isBgExtraction = 1
    return res

def handDetection(contour, drawing):
    # hand detection by calculating the convexy hull and convexity defects
    hull = cv2.convexHull(contour, returnPoints = False)

    if len(hull) > 3:
        cvxDef = cv2.convexityDefects(contour, hull)
        if type(cvxDef) != type(None):
            count = 0
            for i in range(cvxDef.shape[0]):
                start_of_cvxDef, end_of_cvxDef, far_of_cvxDef, depth_of_cvxDef = cvxDef[i][0]
                # the start point, end point and deepest point forms a triangle, if the bottom angle is less than
                # 90 degrees, there are two fingers alongside
                start = tuple(contour[start_of_cvxDef][0])
                end = tuple(contour[end_of_cvxDef][0])
                far = tuple(contour[far_of_cvxDef][0])

                v1 = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                v2 = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                v3 = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                angle = math.acos((v2 ** 2 + v3 ** 2 - v1 ** 2) / (2 * v2 * v3)) # calculate the consine value
                if angle < math.pi / 2:
                    #if bottom angle < 90, there is a finger alongside
                    count += 1
                    #cv2.line(img, start, end, (0, 255, 0), 2) # draw convexy hull
                    cv2.circle(drawing, far, 6, (255, 0, 0), -1) # draw bottom points
            return True, count 
    return False, 0 

def printThreshold(threshold):
    # print the dynamic threshold value
    print('Changing threshold to : ', str(threshold))

# Real-time computer vision and camera operation
cameraIndex = int(input('Please specify which camera to use, 0 for default webcam: '))
camera = cv2.VideoCapture(cameraIndex)
camera.set(10, 200) # set the brightness
font = cv2.FONT_HERSHEY_SIMPLEX # set the font

# threshold window
cv2.namedWindow('Threshold Settings')
cv2.createTrackbar('Threshold', 'Threshold Settings', threshold, 100, printThreshold)

#keyboard operation commands
print('Press "b" to subtract background or press "r" to reset the extraction.')

while camera.isOpened():
    success, frame = camera.read()
    threshold = cv2.getTrackbarPos('Threshold', 'Threshold Settings')
    frame = cv2.bilateralFilter(frame, 5, 50, 100) # bilateral filter
    frame = cv2.flip(frame, 1) # mirror the frame
    # define ROI
    cv2.rectangle(frame, (int(roi_x * frame.shape[1]), 0), (frame.shape[1], int(roi_y * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('Camera', frame)

    # background subtraction
    if bg_extraction == 1:
        img = removeBG(frame)
        img = img[0: int(roi_y * frame.shape[0]), int(roi_x * frame.shape[1]): frame.shape[1]] # extract ROI
        cv2.imshow('Mask', img) # mask image 

        # gaussian blur
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurImg = cv2.GaussianBlur(grayImg, (blur, blur), 0)
        cv2.imshow('Blur', blurImg) # blur image

        # dynamically adjust the threshold value by detecting the light condition, get the intensity value of 
        # top center pixel
        w, h = np.shape(img)[: 2]
        lightLv = grayImg[int(h / 100)][int(w / 2)]
        thresLv = bg_threshold + lightLv
        print('Threshold changes to :', thresLv)
        success, thresImg = cv2.threshold(blurImg, thresLv, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('Threshold Image', thresImg) # dynamic threshold image

        # contour of hand
        newThresImg = copy.deepcopy(thresImg)
        _, cnt, hierarchy = cv2.findContours(newThresImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

            hull = cv2.convexHull(maxCnt)
            drawing = np.zeros(img.shape, np.uint8) 
            moment = cv2.moments(maxCnt)
            if moment['m00'] != 0:
                #center coordinates, x = M10/M00, y = M01/M00
                center_x = int(moment['m10'] / moment['m00'])
                center_y = int(moment['m01'] / moment['m00'])
                center = (center_x, center_y)
            #cv2.circle(img, center, 5, (0, 0, 255), 2)
            cv2.drawContours(drawing, [maxCnt], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 3)

            isFinger, count = handDetection(maxCnt, drawing)
            # finger counting
            if checkFinger is True:
                if isFinger is True:
                    print('Finger count = ', count)

        cv2.imshow('Drawing', drawing)

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
        # print the drawing
        h, w = img.shape[: 2]
        printImg = cv2.resize(img, (h, w), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('Data/Output/drawing.jpg', drawing)
        print('Drawing has been printed.')


#=================================test code===================================
# a = np.array([(1, 2), (3, 4)])
# for i in range(a.shape[0]):
#   b, c = a[i][0]
#   print(b,c)
# print(a, a.shape)
# a = [1, 2, 3, 4, ['a', 'b']]
# b = a 
# print('b = ', b)
# c = copy.copy(a)
# print('c = ', c)
# d = copy.deepcopy(a)
# print('d = ', d)
# a[0] = 'hhhhh'
# a[4][0] = 'test'
# print(a, c, d)