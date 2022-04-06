import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Tutorial https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

i = 11
query_image = 'lusitania_{}.png'.format(i)
train_image = 'lusitania_{}.png'.format(i)
img1 = cv2.imread('images/templates/{}'.format(query_image), 0)
img2 = cv2.imread('images/original/{}'.format(train_image), 0)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        good.append([m])

draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matchesMask = None,flags = 2)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, **draw_params)
#cv.imwrite('ex3_4_result/result.png', img3)
plt.imshow(img3)
plt.show()
